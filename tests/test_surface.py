import os
import numpy as np
from vmecpp.simsopt_compat import Vmec
from alpha_opt import init_optimizable_surface, SurfaceGarabedianQuantiles, DATA_DIR


def test_init_optimizable_surface():
    nfp = 3
    major_radius = 2.3
    minor_radius = 0.5
    for mn_max in [1, 2, 3]:
        surface, dim_x, x_scale, x0 = init_optimizable_surface(
            mn_max, mn_max, nfp, major_radius, minor_radius
        )


def test_surface_Garabedian_quantiles():
    nfp = 3
    major_radius = 3.3
    minor_radius = 0.2

    surface = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=1,
        ntor=1,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    np.testing.assert_equal(len(surface.x), 7)
    np.testing.assert_allclose(
        surface.to_RZFourier().major_radius(), major_radius, rtol=0.007
    )
    np.testing.assert_allclose(
        surface.to_RZFourier().minor_radius(), minor_radius, rtol=0.12
    )

    surface.x = np.ones_like(surface.x) * 0.6
    np.testing.assert_equal(len(surface.x), 7)
    np.testing.assert_allclose(
        surface.to_RZFourier().major_radius(), major_radius, rtol=0.011
    )
    np.testing.assert_allclose(
        surface.to_RZFourier().minor_radius(), minor_radius, rtol=0.1
    )

    surface2 = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=2,
        ntor=3,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    np.testing.assert_equal(len(surface2.x), 5 * 7 - 2)


def test_surface_Garabedian_quantiles_exact_radii():
    nfp = 3
    major_radius = 2.3
    minor_radius = 0.5

    surface = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=2,
        ntor=3,
        major_radius=major_radius,
        minor_radius=minor_radius,
        exact_radii=True,
    )
    surface2 = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=2,
        ntor=3,
        major_radius=major_radius,
        minor_radius=minor_radius,
        exact_radii=False,
    )

    # Perturb controls so exact_radii enforcement is exercised on recompute.
    surface.x = np.ones_like(surface.x) * 0.6
    surface2.x = surface.x
    rz_surface = surface.to_RZFourier()
    rz_surface2 = surface2.to_RZFourier()
    print("Final major radius:", rz_surface.major_radius(), "minor radius:", rz_surface.minor_radius())

    np.testing.assert_allclose(
        rz_surface.major_radius(), major_radius, atol=1e-10, rtol=1e-12
    )
    np.testing.assert_allclose(
        rz_surface.minor_radius(), minor_radius, atol=1e-10, rtol=1e-12
    )
    # Surfaces should be identical up to the overall scale and the major radius
    np.testing.assert_allclose(
        rz_surface.x[1:] * rz_surface2.minor_radius() / rz_surface.minor_radius(),
        rz_surface2.x[1:],
    )


def test_surface_Garabedian_quantiles_regression():
    """Compare to 20260306-02_weightedQuantile_on_hdf5_Garabedian_interactive.py"""
    nfp = 3
    minor_radius = 0.2
    major_radius = 10.0 * minor_radius

    surface = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=1,
        ntor=1,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    reference_x = np.array([-9.58024190e-02, 1.00000000e+00, 1.04883385e-01, 4.40607174e-02,
                            1.00000000e+01, 7.56137942e-01, -7.48472419e-04, -8.59236577e-02,
                            -4.71235395e-01])

    np.testing.assert_allclose(surface.surface_garabedian.x, reference_x * minor_radius, rtol=1e-8)


def test_surface_Garabedian_quantiles_with_vmec():
    vmec = Vmec(os.path.join(DATA_DIR, "input.vmec"))

    nfp = 3
    major_radius = 2.3
    minor_radius = 0.5

    surface = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=2,
        ntor=3,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    vmec.boundary = surface
    vmec.indata.nfp = (
        nfp  # Vmec++ does not automatically get nfp from the boundary surface!
    )

    vmec.run()
    np.testing.assert_equal(vmec.wout.nfp, nfp)
    np.testing.assert_allclose(vmec.wout.Rmajor_p, major_radius, rtol=0.02)
    np.testing.assert_allclose(vmec.wout.Aminor_p, minor_radius, rtol=0.1)
    print("iota:", list(float(x) for x in vmec.wout.iotaf))
    np.testing.assert_allclose(vmec.wout.iotaf[0], 0.7235724832954784)
    np.testing.assert_allclose(vmec.wout.iotaf[-1], 0.7833059615881733)
