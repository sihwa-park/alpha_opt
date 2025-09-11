#!/usr/bin/env python

import numpy as np
from vmecpp.simsopt_compat import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual

from alpha_opt.surface import init_optimizable_surface
from alpha_opt.gen_f import nlopt_gen_f
from alpha_opt.sim_f import minimal_sim_f
from alpha_opt.objective import get_objective
from alpha_opt.libE import libE_driver_serial

aspect_ratio = 8.0
major_radius = 1.0
minor_radius = major_radius / aspect_ratio

# Maximum mode numbers to vary in the SurfaceRZFourier representation.
m_max = 2
n_max = 2

initial_step_size = 1e-4

np.set_printoptions(linewidth=300)

vmec = Vmec("input.vmec")

surface, dim_x, x_scale, x0 = init_optimizable_surface(
    m_max,
    n_max,
    vmec.indata.nfp,
    major_radius,
    minor_radius,
)

qs = QuasisymmetryRatioResidual(
    vmec, 
    np.linspace(0, 1, 11),
    helicity_m=1,
    helicity_n=1,
)

wrapped_objective = get_objective(vmec, surface, x_scale, qs.total, fail_val=1000.0)
# Options for minimal_sim_f:
sim_user_specs = {"objective": wrapped_objective}

import pickle
pickle.dumps(sim_user_specs)  # Test that minimal_sim_f can be pickled.

# Options for nlopt_gen_f:
gen_user_specs = {
    "algorithm": "LN_BOBYQA",
    "x0": x0,
    "initial_step_size": initial_step_size,
}

if __name__ == "__main__":  # Python-quirk required on macOS and windows
    libE_driver_serial(minimal_sim_f, sim_user_specs, nlopt_gen_f, gen_user_specs, dim_x, 5)
