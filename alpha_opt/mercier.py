import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.constants import mu_0

def get_DMerc_normalized(vmec):
    """
    Return the normalized version of DMerc which I defined in
    20230124-01 Normalizations for Mercier objective.lyx

    The returned array is defined on the full vmec s grid.

    Parameters
    ----------
    vmec : Vmec
        A simsopt or vmec++ Vmec object that has already been run.

    Returns
    -------
    DMerc_normalized : array_like
        The Mercier stability criterion DMerc, normalized as described in the note.
    """
    s = vmec.s_full_grid
    s_safe = s.copy()
    s_safe[0] = 1e-15  # To avoid division by 0
    pressure = InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.presf)
    d_pressure_d_s = pressure.derivative()(s)

    G = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bvco[1:])(s)

    volume = vmec.wout.volume_p
    Phi = vmec.wout.phi[-1]

    normalization = np.abs(mu_0 * volume * G * d_pressure_d_s / (2 * (Phi**4) * s_safe * vmec.wout.volavgB))

    degree = 1
    DMerc = InterpolatedUnivariateSpline(vmec.s_full_grid[1:-1], vmec.wout.DMerc[1:-1], k=degree)(s)
    return DMerc / normalization


def get_worst_DMerc_normalized(vmec, s_min, s_max):
    """
    Get the worst (most negative) value of DMerc_normalized in the range s_min <= s <= s_max.

    Parameters
    ----------
    vmec : Vmec
        A simsopt or vmec++ Vmec object that has already been run.
    s_min : float
        Minimum value of s to consider.
    s_max : float
        Maximum value of s to consider.

    Returns
    -------
    worst_DMerc : float
        The most negative value of DMerc_normalized in the specified range.
    """
    s = vmec.s_full_grid
    DMerc_normalized = get_DMerc_normalized(vmec)
    mask = (s >= s_min) & (s <= s_max)
    return np.min(DMerc_normalized[mask])
