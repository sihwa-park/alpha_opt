import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import elementary_charge
import vmecpp
from alpha_opt import DATA_DIR
from alpha_opt.profiles import n_m3_func, T_keV_func, DT_reaction_rate

def test_pressure_profile_matches_vmec():
    """Ensure that the profiles in python match the pressure profile in VMEC."""
    filename = os.path.join(DATA_DIR, "input.finite_beta")
    vmec_input = vmecpp.VmecInput.from_file(filename)
    vmec_output = vmecpp.run(vmec_input)
    ns = vmec_output.wout.ns
    pressure_vmec = vmec_output.wout.pres[1:]
    s_full = np.linspace(0, 1, ns)
    ds = s_full[1] - s_full[0]
    s_half = s_full[1:] - 0.5 * ds
    n_m3_profile = n_m3_func(s_half)
    T_keV_profile = T_keV_func(s_half)
    pressure_should_be = 2 * n_m3_profile * T_keV_profile * 1e3 * elementary_charge  # in Pa

    # plt.plot(s_half, pressure_vmec, label="VMEC")
    # plt.plot(s_half, pressure_should_be, label="Should Be")
    # plt.xlabel("s")
    # plt.ylabel("Pressure (Pa)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    np.testing.assert_allclose(pressure_vmec, pressure_should_be, rtol=2e-13)

def test_reaction_rate():
    """Compare the DT reaction rate to the function coded up by Michael C."""

    s = np.linspace(0, 1, 100)[:-1]
    def s_density(s):
        return (
            ((1 - s**5) ** 2)
            * ((1 - s) ** (-2 / 3))
            * np.exp(-19.94 * (12 * (1 - s)) ** (-1 / 3))
        )
    reaction_rate_Michael = s_density(s)

    ni = 1 - s**5
    Ti_keV = 12 * (1 - s)
    reaction_rate_this_code = DT_reaction_rate(ni, Ti_keV)
    # Don't worry about the constant in front, just make sure the ratio is constant
    ratio = reaction_rate_Michael / reaction_rate_this_code
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-14)