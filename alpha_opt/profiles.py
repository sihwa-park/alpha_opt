import numpy as np
from scipy.constants import elementary_charge
import matplotlib.pyplot as plt

def n_m3_func(s, n0=3e20):
    return n0 * (1 - 2 * s**4 + 1.2 * s**6)

def T_keV_func(s, T0=15):
    return T0 * (1 - 2 * s + 2 * s**2 - s**3)

def DT_reaction_rate(ni, Ti_keV):
    """DT fusion reaction rate from NRL plasma formulary, page 45 (2023 edition).
    """
    return ni * ni * (3.68e-12) * (Ti_keV**(-2.0/3.0) * np.exp(-19.94 * (Ti_keV**(-1.0/3.0))))

"""Compute the relative DT reaction rate for our profiles, normalized to the value at s=0."""
def relative_DT_reaction_rate_for_our_profiles(s):
    ni = n_m3_func(s) / 2
    ni0 = n_m3_func(0) / 2
    Ti_keV = T_keV_func(s)
    Ti0_keV = T_keV_func(0)
    return DT_reaction_rate(ni, Ti_keV) / DT_reaction_rate(ni0, Ti0_keV)

"""Using rejection sampling, sample s according to the alpha birth profile."""
def sample_alpha_birth_s():
    bound = 1.0
    x = np.random.uniform()
    y = bound * np.random.uniform()

    while relative_DT_reaction_rate_for_our_profiles(x) < y:
        assert relative_DT_reaction_rate_for_our_profiles(x) <= bound
        x = np.random.uniform()
        y = bound * np.random.uniform()
    return x


def plot_profiles(output_path="profiles_for_alpha_opt.pdf"):
    """Plot key profiles on a 1x4 grid and save to a PDF file."""
    rho = np.linspace(0.0, 1.0, 200)
    s = rho**2

    n_vals = n_m3_func(s)
    T_vals = T_keV_func(s)
    pressure_vals = 2.0 * n_vals * (T_vals * 1000 * elementary_charge) / 1e6  # Convert keV to Joules and then to MPa
    rel_rate_vals = relative_DT_reaction_rate_for_our_profiles(s)

    # fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))

    y_data = [n_vals, T_vals, pressure_vals, rel_rate_vals]
    y_labels = [
        "$n_e$ [m$^{-3}$]",
        "$T_i = T_e$ [keV]",
        "Pressure [MPa]",
        "$\langle \sigma v \\rangle \; / \; \langle \sigma v \\rangle(\\rho=0)$",
    ]
    titles = [
        "Density",
        "Temperature",
        "Pressure",
        "Relative DT fusion reaction rate",
    ]

    colors = ['g', 'r', 'b', 'm']
    for ax, y, ylabel, title, color in zip(axes, y_data, y_labels, titles, colors):
        ax.plot(s, y, color=color)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Normalized minor radius $\\rho$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.048, bottom=0.145, right=0.986, top=0.926, wspace=0.28)
    fig.savefig(output_path, format="pdf")
    plt.show()