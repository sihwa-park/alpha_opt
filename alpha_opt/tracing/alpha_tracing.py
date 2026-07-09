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
    from firm3d.catapult.utils import boozer_interpolant
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

def sample_stz_s_0p25(field, J_max):
    s = 0.25
    theta, zeta = sample_tz(s, J_max, field)
    return np.array([s, theta, zeta])

def generate_interpolant_and_initial_conditions(
    wout_filename,
    mbooz=12,
    nbooz=12,
    n_particles=5000,
    vacuum=False,
    profiles="realistic",
    seed=8,
):
    """
    Generate initial conditions for alpha particles.

    If profiles is "firm3d", use firm3d's sampling function for s.
    If profiles is "realistic", use alpha_opt's sampling function for s, which assumes different
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
        field, nfp, n_metagrid_pts, n_metagrid_pts, n_metagrid_pts, vacuum=vacuum
    )
    print(f"Time for boozer_interpolant(): {time.time()-t2:.3f} s", flush=True)

    # Evaluate error in interpolation
    print("Error in |B| interpolation", field.estimate_error_modB(1000), flush=True)

    # set seed for consistency (override with `seed` to resample particle ICs)
    np.random.seed(seed)

    print("About to create stz_inits. maxJ=", maxJ)
    t3 = time.time()
    if profiles == "firm3d":
        sample_func = firm3d_sample_stz
    elif profiles == "realistic":
        sample_func = sample_stz
    elif profiles == "0.25":
        sample_func = sample_stz_s_0p25
    else:
        raise ValueError(f"Invalid value for profiles: {profiles}")
    
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
    profiles="realistic",
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
            profiles,
        )
    )

    with open("start.dat", "w") as f:
        for i in range(n_particles):
            f.write(
                f"{stz_inits[i,0]:24.15g} {stz_inits[i,1]:24.15g} {stz_inits[i,2]:24.15g} 1.0 {vpar_inits[i] / ALPHA_BIRTH_SPEED:24.15g}\n"
            )


def trace_catapult(
    quad_info,
    stz_inits,
    parallel_speeds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    srange,
    trange,
    zrange,
    psi0,
    t_block,
    maxloss,
):
    total_n = stz_inits.shape[0]
    n_particles = total_n
    current_time = np.zeros(n_particles)

    dt = -np.ones(n_particles)
    mu = -np.ones(n_particles)

    # convert Boozer to pseudo-Cartesian coordinates
    s = stz_inits[:, 0]
    theta = stz_inits[:, 1]
    x1 = s*np.cos(theta)
    x2 = s*np.sin(theta)

    stz_inits[:, 0] = x1
    stz_inits[:, 1] = x2
    stz_inits = np.ascontiguousarray(stz_inits)

    # when we filter particles out for leaving
    # we need to remember their original index
    ids = np.arange(n_particles, dtype=int)
    loss_times = [-1.0 for i in range(n_particles)]

    n_steps = int(tmax / t_block)
    for step in range(n_steps):
        # keep track of the tmax we will reach at the end of the loop
        # each particle needs to advance to step_end_time
        local_tmax = np.maximum((step + 1) * t_block - current_time, 0.0)

        # advance particles to step_end_time
        dt = np.ascontiguousarray(dt)
        local_tmax = np.ascontiguousarray(local_tmax)
        mu = np.ascontiguousarray(mu)

        step_data = firm3dpp.boozer_gpu_tracing(
            quad_pts=quad_info,
            srange=srange,
            trange=trange,
            zrange=zrange,
            stz_init=stz_inits.copy(),
            m=mass,
            q=charge,
            vtotal=vtotal,
            vtang=parallel_speeds.copy(),
            tmax=local_tmax,
            tol=tol,
            dt_in=dt,
            mu_in=mu,
            psi0=psi0,
            nparticles=n_particles,
            vacuum=False
        )
        print("finished tracing")
        step_data = np.reshape(step_data, (n_particles, 7))

        dt = step_data[:, 5].copy()
        mu = step_data[:, 6].copy()

        # compute new current time for each particle
        step_data[:, 0] += current_time
        current_time = step_data[:, 0]

        # store data using stored indices
        for i, idx in enumerate(ids):
            if local_tmax[i] > 0.0:
                loss_times[idx] = current_time[i]

        # find lost particles
        s_end = np.sqrt(step_data[:, 1]**2 + step_data[:, 2]**2)
        idx_keep = (current_time < tmax) & (s_end < 1.0)



        # remove lost particles
        stz_inits = step_data[idx_keep, 1:4].copy()
        parallel_speeds = step_data[idx_keep, 4].copy()
        ids = ids[idx_keep]
        current_time = current_time[idx_keep]
        dt = dt[idx_keep].copy()
        mu = mu[idx_keep].copy()

        n_particles = stz_inits.shape[0]

        # if energy losses have exceeded maxloss, return the time at which this occurred
        energy_loss_fraction = np.sum([np.exp(-loss_times[i] / tmax) for i in range(len(loss_times)) if i not in ids]) / total_n
        if(energy_loss_fraction >= maxloss):
            print(f"Energy loss fraction {energy_loss_fraction} exceeded maxloss {maxloss}. Raw loss frac: {1-len(ids)/total_n} Returning early.")
            return loss_times
    
    return loss_times


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
    profiles="realistic",
    seed=8,
):
    """
    If maxloss is >= 1, this function returns the energy loss fraction at t_max.
    If maxloss < 1, it returns the time at which the energy loss fraction exceeds maxloss.

    `seed` controls only the particle initial-condition sampling (see
    `generate_interpolant_and_initial_conditions`), not the VMEC equilibrium;
    it defaults to 8, matching the previously-hardcoded value.
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
            profiles,
            seed,
        )
    )
    print("tracing particles", flush=True)

    # trace on GPU
    tracing_start = time.time()
    vtotal = np.sqrt(2*ENERGY / MASS)
    loss_times = trace_catapult(
        quad_info,  
        stz_inits,
        vpar_inits,
        t_max,
        MASS,
        CHARGE,
        vtotal,
        tol,
        srange,
        trange,
        zrange,
        field.psi0,
        t_block=t_block,
        maxloss=maxloss
    )
    print(f"Tracing took {time.time() - tracing_start}\n")

    return alpha_loss_objective_from_times(
        loss_times, tau, maxloss, t_max
    )