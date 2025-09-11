from alpha_opt import init_optimizable_surface

def test_init_optimizable_surface():
    nfp = 3
    major_radius = 2.3
    minor_radius = 0.5
    for mn_max in [1, 2, 3]:
        surface, dim_x, x_scale, x0 = init_optimizable_surface(mn_max, mn_max, nfp, major_radius, minor_radius)