#!/usr/bin/env python


import logging
import math
import time
from math import sqrt
import numpy as np
from scipy.io import netcdf_file
import pandas as pd
from booz_xform import Booz_xform

from simsopt._core import Optimizable
from simsopt.util.constants import (
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
    ALPHA_PARTICLE_CHARGE as CHARGE,
)

try:
    import firm3d
except ImportError:
    firm3d = None

if firm3d is None:
    from simsopt.field import BoozerRadialInterpolant, InterpolatedBoozerField
else:
    from firm3d.field.boozermagneticfield import (
        BoozerRadialInterpolant,
        InterpolatedBoozerField,
    )
    from firm3d.util.gpu_utils import boozer_interpolant
    import firm3dpp
    # from firm3d.util.sampling import sample_stz

from .profiles import sample_alpha_birth_s
from .loss_times import compute_energy_loss_fraction, time_at_which_energy_loss_exceeds

# logging.basicConfig()
# logger = logging.getLogger('simsopt.field.tracing')

# Initialize vpar
ALPHA_BIRTH_SPEED = np.sqrt(2 * ENERGY / MASS)


# Sample theta, zeta for a given s via rejection sampling
# This function differs from the one in firm3d in that abs() is applied to J.
def sample_tz(s, J_max, field):
    J = rand_J = 0
    while rand_J >= J:
        theta = np.random.uniform(low=0, high=2 * math.pi, size=1)
        zeta = np.random.uniform(low=0, high=2 * math.pi, size=1)
        rand_J = np.random.uniform(low=0, high=abs(J_max), size=1)

        loc = np.array([s, theta[0], zeta[0]]).reshape(1, 3)
        field.set_points(loc)

        G = field.G()
        iota = field.iota()
        I = field.I()
        modB = field.modB()
        J = abs(G + iota * I) / (modB**2)
        J = J[0][0]
        assert J <= abs(J_max)
    return theta[0], zeta[0]


# Sample s,t,z
def sample_stz(field, J_max):
   s = sample_alpha_birth_s()
   theta, zeta = sample_tz(s, J_max, field)
   return np.array([s, theta, zeta])


class AlphaLosses(Optimizable):
    """
    Compute alpha particle losses in a stellarator magnetic field using GPU-accelerated particle tracing.

    This class implements an objective function for stellarator optimization that calculates
    the fraction of alpha particles that are lost from the plasma. Alpha particles are born
    with a specific energy distribution and are traced through the magnetic field using
    Boozer coordinates until they either leave the plasma or reach the maximum simulation time.

    The class uses GPU-accelerated particle tracing via simsoptpp for efficient computation
    of large numbers of particle trajectories. The magnetic field is represented using
    interpolated Boozer coordinates for fast evaluation during tracing.

    Args:
        booz: A Booz_xform object representing the magnetic equilibrium in Boozer coordinates.
        n_particles (int, optional): Number of alpha particles to trace. Defaults to 25000.
        t_max (float, optional): Maximum simulation time in seconds. Particles that don't
            leave the plasma within this time are considered confined. Defaults to 1e-2.
        tau (float, optional): Time constant parameter (currently unused). Defaults to 1e-1.

    Attributes:
        booz: The Boozer coordinate representation of the magnetic field.
        n_particles (int): Number of particles to trace in each evaluation.
        t_max (float): Maximum simulation time.
        tau (float): Collisional slowing-down time, used to discount losses at late times.

    Returns:
        The objective function J() returns a numpy array containing the loss fraction,
        which is the fraction of traced particles that leave the plasma before t_max.

    Example:
        >>> from booz_xform import Booz_xform
        >>> booz = Booz_xform()
        >>> booz.read_boozmn("boozmn.nc")
        >>> alpha_losses = AlphaLosses(booz, n_particles=10000, t_max=5e-3)
        >>> loss_fraction = alpha_losses.J()
    """

    def __init__(self, booz, n_particles=25000, t_max=1e-2, tau=1e-1):
        self.booz = booz
        self.n_particles = n_particles
        self.t_max = t_max
        self.tau = tau
        super().__init__(depends_on=[booz])

    def J(self):
        self.booz.run()

        # Compute VMEC equilibrium
        t1 = time.time()
        # equil = Booz_xform()
        # equil.verbose = 0
        # equil.read_boozmn(filename)
        # N = -4

        order = 3
        # bri = BoozerRadialInterpolant(equil, order, no_K=True, N=N)
        bri = BoozerRadialInterpolant(self.booz.bx, order, no_K=True)

        nfp = self.booz.nfp
        degree = 3
        n_metagrid_pts = 15
        field = InterpolatedBoozerField(
            bri,
            degree,
            ns_interp=n_metagrid_pts,
            ntheta_interp=n_metagrid_pts,
            nzeta_interp=n_metagrid_pts,
        )
        srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
            field, nfp, n_metagrid_pts
        )

        # Evaluate error in interpolation
        print("Error in |B| interpolation", field.estimate_error_modB(1000), flush=True)

        # set seed for consistency
        np.random.seed(8)

        stz_inits = np.vstack(
            [sample_stz(field, maxJ) for i in range(self.n_particles)]
        )
        vpar_inits = ALPHA_BIRTH_SPEED * np.random.uniform(
            low=-1, high=1, size=self.n_particles
        )

        print("tracing particles")

        # trace on GPU
        last_time = firm3dpp.boozer_gpu_tracing(
            quad_pts=quad_info,
            srange=srange,
            trange=trange,
            zrange=zrange,
            stz_init=stz_inits,
            m=MASS,
            q=CHARGE,
            vtotal=ALPHA_BIRTH_SPEED,
            vtang=vpar_inits,
            tmax=self.t_max,
            tol=1e-9,
            psi0=field.psi0,
            nparticles=self.n_particles,
        )

        last_time = np.reshape(last_time, (self.n_particles, 5))
        particle_data = pd.DataFrame(
            {
                "s_start": stz_inits[:, 0],
                "t_start": stz_inits[:, 1],
                "z_start": stz_inits[:, 2],
                "vpar_start": vpar_inits,
                "s_end": last_time[:, 1],
                "t_end": last_time[:, 2],
                "z_end": last_time[:, 3],
                "vpar_end": last_time[:, 3],
                "last_time": last_time[:, 0],
            }
        )

        particle_data.to_csv("particle_data.csv")

        did_leave = [t < self.t_max for t in particle_data["last_time"]]
        loss_frac = sum(did_leave) / len(did_leave)
        print(f"Number of particles= {self.n_particles}")
        print(f"Number loss fraction: {loss_frac:.3f}")

        energy_loss_fraction = compute_energy_loss_fraction(
            particle_data["last_time"], self.tau
        )
        print(f"Energy loss fraction at end of tracing: {energy_loss_fraction:.3f}")
        return energy_loss_fraction


def compute_alpha_loss(
    wout_filename,
    mbooz=12,
    nbooz=12,
    n_particles=25000,
    t_max=1e-2,
    tau=1e-1,
    min_dt=1e-10,
    maxloss=10.0,
    t_block=1e-4,
):
    """
    If maxloss is >= 1, this function returns the energy loss fraction at t_max.
    If maxloss < 1, it returns the time at which the energy loss fraction exceeds maxloss.
    """
    print(
        f"Computing alpha losses with {n_particles} particles, t_max={t_max}, tau={tau}"
    )

    with netcdf_file(wout_filename, "r") as f:
        nfp = int(f.variables["nfp"][()])

    # # Compute VMEC equilibrium
    # equil = Booz_xform()
    # equil.verbose = 0
    # equil.read_boozmn("../boozmn_QH_boots.nc")
    # nfp = equil.nfp
    # N = -4

    order = 3
    # N = None
    # bri = BoozerRadialInterpolant(equil, order, no_K=True, N=N)
    t1 = time.time()
    bri = BoozerRadialInterpolant(
        wout_filename,
        order,
        mpol=mbooz,
        ntor=nbooz,
        no_K=True,
        write_boozmn=False,
        verbose=0,
    )
    # bri = BoozerRadialInterpolant(equil, order, mpol=mbooz, ntor=nbooz, no_K=True, write_boozmn=False, verbose=1, N=N)
    print(
        f"Time to initialize BoozerRadialInterpolant: {time.time()-t1:.3f} s",
        flush=True,
    )

    degree = 3
    n_metagrid_pts = 15
    t1 = time.time()
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=n_metagrid_pts,
        ntheta_interp=n_metagrid_pts,
        nzeta_interp=n_metagrid_pts,
    )
    print(
        f"Time to initialize InterpolatedBoozerField: {time.time()-t1:.3f} s",
        flush=True,
    )
    t2 = time.time()
    srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
        field, nfp, n_metagrid_pts
    )
    print(f"Time for boozer_interpolant(): {time.time()-t2:.3f} s", flush=True)

    # Evaluate error in interpolation
    print("Error in |B| interpolation", field.estimate_error_modB(1000), flush=True)

    # set seed for consistency
    np.random.seed(8)

    print("About to create stz_inits. maxJ=", maxJ)
    stz_inits = np.vstack([sample_stz(field, maxJ) for i in range(n_particles)])
    print("Finished creating stz_inits", flush=True)
    vpar_inits = ALPHA_BIRTH_SPEED * np.random.uniform(low=-1, high=1, size=n_particles)

    print("tracing particles", flush=True)

    # trace on GPU
    last_time = firm3dpp.boozer_gpu_tracing(
        quad_pts=quad_info,
        srange=srange,
        trange=trange,
        zrange=zrange,
        stz_init=stz_inits,
        m=MASS,
        q=CHARGE,
        vtotal=ALPHA_BIRTH_SPEED,
        vtang=vpar_inits,
        tmax=t_max,
        tol=1e-9,
        psi0=field.psi0,
        nparticles=n_particles,
        min_dt=min_dt,
        maxloss=maxloss,
        t_block=t_block,
    )

    last_time = np.reshape(last_time, (n_particles, -1))
    particle_data = pd.DataFrame(
        {
            "s_start": stz_inits[:, 0],
            "t_start": stz_inits[:, 1],
            "z_start": stz_inits[:, 2],
            "vpar_start": vpar_inits,
            "s_end": last_time[:, 1],
            "t_end": last_time[:, 2],
            "z_end": last_time[:, 3],
            "vpar_end": last_time[:, 3],
            "last_time": last_time[:, 0],
        }
    )
    particle_data.to_csv("particle_data.csv")

    did_leave = [t < t_max for t in particle_data["last_time"]]
    loss_frac = sum(did_leave) / len(did_leave)

    energy_loss_fraction = compute_energy_loss_fraction(particle_data["last_time"], tau)
    print(f"n_particles: {n_particles}, number loss fraction: {loss_frac:.3f}"
        f", energy loss fraction at end of tracing: {energy_loss_fraction:.3f}")

    if maxloss < 1.0:
        t_exceed = time_at_which_energy_loss_exceeds(
            particle_data["last_time"], tau, maxloss, t_max
        )
        print(f"Time at which energy loss fraction exceeds {maxloss}: {t_exceed:.6e} s")
        return t_exceed

    return energy_loss_fraction
