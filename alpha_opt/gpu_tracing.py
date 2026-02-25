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
    from firm3d.util.sampling import sample_stz as firm3d_sample_stz
    import firm3dpp

    # from firm3d.util.sampling import sample_stz

from .profiles import sample_alpha_birth_s
from .loss_times import alpha_loss_objective_from_times

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


def generate_interpolant_and_initial_conditions(
    wout_filename,
    mbooz=12,
    nbooz=12,
    n_particles=5000,
    vacuum=False,
    firm3d_profiles=False,
):
    """
    Generate initial conditions for alpha particles.

    If firm3d_profiles is True, use firm3d's sampling function for s.
    If False, use alpha_opt's sampling function for s, which assumes different
    density and temperature profiles.
    """
    start_time = time.time()
    
    with netcdf_file(wout_filename, "r") as f:
        nfp = int(f.variables["nfp"][()])

    order = 3
    # N = None
    # bri = BoozerRadialInterpolant(equil, order, no_K=True, N=N)
    t1 = time.time()
    bri = BoozerRadialInterpolant(
        wout_filename,
        order,
        mpol=mbooz,
        ntor=nbooz,
        no_K=vacuum,
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
        field, nfp, n_metagrid_pts, vacuum=vacuum
    )
    print(f"Time for boozer_interpolant(): {time.time()-t2:.3f} s", flush=True)

    # Evaluate error in interpolation
    print("Error in |B| interpolation", field.estimate_error_modB(1000), flush=True)

    # set seed for consistency
    np.random.seed(8)

    print("About to create stz_inits. maxJ=", maxJ)
    t3 = time.time()
    if firm3d_profiles:
        sample_func = firm3d_sample_stz
    else:
        sample_func = sample_stz
    stz_inits = np.vstack([sample_func(field, maxJ) for i in range(n_particles)])
    print(f"Finished creating stz_inits. Took {time.time()-t3:.3f} s", flush=True)
    vpar_inits = ALPHA_BIRTH_SPEED * np.random.uniform(low=-1, high=1, size=n_particles)

    print("Total time in generate_interpolant_and_initial_conditions:", time.time() - start_time)
    return stz_inits, vpar_inits, srange, trange, zrange, quad_info, field


def write_simple_start(
    wout_filename,
    mbooz=12,
    nbooz=12,
    n_particles=5000,
    vacuum=False,
    firm3d_profiles=False,
):
    """
    Write a start.dat file for SIMPLE with particle initial conditions.
    """
    stz_inits, vpar_inits, srange, trange, zrange, quad_info, field = (
        generate_interpolant_and_initial_conditions(
            wout_filename,
            mbooz,
            nbooz,
            n_particles,
            vacuum,
            firm3d_profiles,
        )
    )

    with open("start.dat", "w") as f:
        for i in range(n_particles):
            f.write(
                f"{stz_inits[i,0]:24.15g} {stz_inits[i,1]:24.15g} {stz_inits[i,2]:24.15g} 1.0 {vpar_inits[i] / ALPHA_BIRTH_SPEED:24.15g}\n"
            )


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
    tol=1e-9,
    vacuum=False,
    firm3d_profiles=False,
):
    """
    If maxloss is >= 1, this function returns the energy loss fraction at t_max.
    If maxloss < 1, it returns the time at which the energy loss fraction exceeds maxloss.
    """
    print(
        f"Computing alpha losses with {n_particles} particles, t_max={t_max}, tau={tau}"
    )

    stz_inits, vpar_inits, srange, trange, zrange, quad_info, field = (
        generate_interpolant_and_initial_conditions(
            wout_filename,
            mbooz,
            nbooz,
            n_particles,
            vacuum,
            firm3d_profiles,
        )
    )
    print("tracing particles", flush=True)

    # trace on GPU
    start_time = time.time()
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
        tol=tol,
        psi0=field.psi0,
        nparticles=n_particles,
        min_dt=min_dt,
        maxloss=maxloss,
        t_block=t_block,
        vacuum=vacuum,
    )
    print("Time to call firm3dpp.boozer_gpu_tracing:", time.time() - start_time)

    t1 = time.time()
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
    print("Time to create DataFrame with particle results:", time.time() - t1)
    t2 = time.time()
    particle_data.to_csv("particle_data.csv")
    print("Time to save particle_data.csv:", time.time() - t2)

    return alpha_loss_objective_from_times(
        particle_data["last_time"], tau, maxloss, t_max
    )[0]
