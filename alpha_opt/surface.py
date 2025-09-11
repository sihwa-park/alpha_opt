import numpy as np
from simsopt.geo import SurfaceRZFourier, SurfaceGarabedian

"""Create an optimizable SurfaceGarabedian.

The aspect ratio, major radius, and minor radius are all fixed.
"""
def init_optimizable_surface(
    m_max,
    n_max,
    nfp,
    major_radius,
    minor_radius,
    elongation=2.0,
    exponential_spectral_scaling_alpha=1.0,
):
    pre_surface = SurfaceRZFourier(
        mpol=m_max,
        ntor=n_max,
        nfp=nfp,
    )
    pre_surface.make_rotating_ellipse(
        major_radius=major_radius,
        minor_radius=minor_radius,
        elongation=elongation,
        torsion=minor_radius,
    )
    # pre_surface.change_resolution(mpol=m_max, ntor=n_max)

    # For Garabedian, don't set mmin = -mmax. Instead center them about 1.
    # surface = SurfaceGarabedian(mmax=2, mmin=0, nmax=1, nmin=-1)
    surface = SurfaceGarabedian.from_RZFourier(pre_surface)
    # surface.change_resolution(mmax=2, mmin=0, nmax=1, nmin=-1)
    print("initial x:", surface.x)
    print(surface.local_dof_names)
    # exit(0)
    surface.set("Delta(1,0)", major_radius) # Set major radius
    surface.set("Delta(0,0)", minor_radius)  # Set minor radius
    surface.fix("Delta(0,0)")  # Minor radius
    surface.fix("Delta(1,0)")  # Major radius
    dim_x = len(surface.x)
    print("x:", surface.x)
    print("dof_names:", surface.dof_names)
    # vmec._should_save_outputs = True  # If you want wout files to be generated.

    # Compute x_scale for the dofs.
    # See ~/work24/20240415-01 Generating random stellarator boundary shapes.docx
    # exponential_spectral_scaling_alpha = 1.0
    ms = []
    ns = []
    for m in range(surface.mmin, surface.mmax + 1):
        for n in range(surface.nmin, surface.nmax + 1):
            if n == 0 and m in [0, 1]:
                continue
            ms.append(m)
            ns.append(n)

    ms = np.array(ms)
    ns = np.array(ns)
    dof_names_should_be = [f"Delta({m},{n})" for m, n in zip(ms, ns)]
    assert surface.local_dof_names == dof_names_should_be, f"Expected {dof_names_should_be}, got {surface.local_dof_names}"
    x_scale = np.exp(-exponential_spectral_scaling_alpha * np.sqrt((ms - 1)**2 + ns**2))
    print("x_scale:", x_scale)

    x0 = surface.x / x_scale

    return surface, dim_x, x_scale, x0
