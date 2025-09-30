import numpy as np

def n_m3_func(s, n0=3e20):
    return n0 * (1 - 2 * s**4 + 1.2 * s**6)

def T_keV_func(s, T0=15):
    return T0 * (1 - 2 * s + 2 * s**2 - s**3)

def DT_reaction_rate(ni, Ti_keV):
    """DT fusion reaction rate from NRL plasma formulary, page 45 (2023 edition).
    """
    return ni * ni * (3.68e-12) * (Ti_keV**(-2.0/3.0) * np.exp(-19.94 * (Ti_keV**(-1.0/3.0))))

def relative_DT_reaction_rate_for_our_profiles(s):
    ni = n_m3_func(s) / 2
    ni0 = n_m3_func(0) / 2
    Ti_keV = T_keV_func(s)
    Ti0_keV = T_keV_func(0)
    return DT_reaction_rate(ni, Ti_keV) / DT_reaction_rate(ni0, Ti0_keV)