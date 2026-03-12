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
