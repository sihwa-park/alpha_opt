# Functions related to the loss times of particles.

import numpy as np

def compute_energy_loss_fraction(times, tau):
    """
    Compute the energy loss fraction of particles.
    """
    t_max = max(times)
    n_particles = len(times)
    numerator = 0.0
    for t in times:
        if t < t_max:
            numerator += np.exp(-t / tau)
    energy_loss_fraction = numerator / n_particles
    return energy_loss_fraction

def time_at_which_energy_loss_exceeds(times, tau, threshold, t_max):
    """
    Determine the time at which the cumulative energy loss fraction exceeds a
    given threshold.
    
    Parameters:
    - times: List or array of particle loss times from gpu_tracing.
    - tau: Characteristic time constant for energy loss.
    - threshold: The energy loss fraction threshold to exceed (between 0 and 1).
    - t_max: This value is returned if the loss fraction never exceeds the threshold.
    """
    n_particles = len(times)
    sorted_times = sorted(times)
    cumulative_energy_loss = 0.0
    for i, t in enumerate(sorted_times):
        cumulative_energy_loss += np.exp(-t / tau)
        energy_loss_fraction = cumulative_energy_loss / n_particles
        if energy_loss_fraction >= threshold:
            return t
    return t_max  # If threshold is never reached