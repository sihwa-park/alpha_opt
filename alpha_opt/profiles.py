def n_m3_func(s, n0=3e20):
    return n0 * (1 - 2 * s**4 + 1.2 * s**6)

def T_keV_func(s, T0=15):
    return T0 * (1 - 2 * s + 2 * s**2 - s**3)