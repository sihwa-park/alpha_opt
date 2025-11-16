import os
import numpy as np
from vmecpp.simsopt_compat import Vmec
from simsopt.mhd.vmec_diagnostics import (
    QuasisymmetryRatioResidual,
    vmec_compute_geometry,
)

from alpha_opt import DATA_DIR
from alpha_opt.surface import init_optimizable_surface
from alpha_opt.objective import get_objective, compute_max_B


def test_objective_failure_after_success():
    aspect_ratio = 8.0
    major_radius = 1.0
    minor_radius = major_radius / aspect_ratio

    # Maximum mode numbers to vary in the SurfaceRZFourier representation.
    m_max = 1
    n_max = 1

    fail_val = 1000.0

    np.set_printoptions(linewidth=300)

    vmec = Vmec(os.path.join(DATA_DIR, "input.vmec"))

    surface, dim_x, x_scale, x0 = init_optimizable_surface(
        m_max,
        n_max,
        vmec.indata.nfp,
        major_radius,
        minor_radius,
    )
    np.testing.assert_equal(dim_x, 7)

    qs = QuasisymmetryRatioResidual(
        vmec,
        np.linspace(0, 1, 11),
        helicity_m=1,
        helicity_n=1,
    )

    wrapped_objective = get_objective(
        vmec, surface, x_scale, qs.total, fail_val=fail_val
    )
    initial_objective = wrapped_objective(x0)
    print("Initial objective:", initial_objective)
    np.testing.assert_almost_equal(initial_objective, 0.21908262722091265)

    # x from ~/Box/work25/20250911-02-libensenble_experiments/20250911-02-004_Ax_without_libE/evals/eval000000
    x = [
        -0.646981954574585,
        -0.2516772747039795,
        0.24202191829681396,
        0.04873645305633545,
        -0.7286624610424042,
        -0.3366534113883972,
        -0.5859023034572601,
    ]
    should_be_a_failure = wrapped_objective(x)
    np.testing.assert_equal(should_be_a_failure, fail_val)


def test_max_B_iteration():
    aspect_ratio = 8.0
    major_radius = 1.0
    minor_radius = major_radius / aspect_ratio

    # Maximum mode numbers to vary in the SurfaceRZFourier representation.
    m_max = 1
    n_max = 1

    fail_val = 1000.0

    np.set_printoptions(linewidth=300)

    vmec = Vmec(os.path.join(DATA_DIR, "input.vmec"))

    surface, dim_x, x_scale, x0 = init_optimizable_surface(
        m_max,
        n_max,
        vmec.indata.nfp,
        major_radius,
        minor_radius,
    )
    np.testing.assert_equal(dim_x, 7)

    qs = QuasisymmetryRatioResidual(
        vmec,
        np.linspace(0, 1, 11),
        helicity_m=1,
        helicity_n=1,
    )

    wrapped_objective1 = get_objective(
        vmec,
        surface,
        x_scale,
        qs.total,
        fail_val=fail_val,
        max_B_iterations=0,
    )
    _ = wrapped_objective1(x0)
    max_B = compute_max_B(vmec)
    print("max B without iteration:", max_B)  # Should be about 30.88
    np.testing.assert_array_less(13, max_B)

    wrapped_objective2 = get_objective(
        vmec,
        surface,
        x_scale,
        qs.total,
        fail_val=fail_val,
        max_B=12,
        max_B_iterations=1,
    )
    _ = wrapped_objective2(x0)
    max_B = compute_max_B(vmec)
    print("max B with iteration:", max_B)
    np.testing.assert_allclose(max_B, 12)
