# Functions related to the loss times of particles.

import numpy as np

def compute_energy_loss_fraction(times, tau):
    """
    Compute the energy loss fraction of particles, as well as the number loss fraction.
    """
    t_max = max(times)
    n_particles = len(times)
    did_leave = [t < t_max for t in times]
    number_loss_fraction = sum(did_leave) / n_particles

    numerator = 0.0
    for t in times:
        # Exclude any particles that made it to the maximum tracing time, since
        # this means they were not lost.
        if t < t_max:
            numerator += np.exp(-t / tau)
    energy_loss_fraction = numerator / n_particles

    return number_loss_fraction, energy_loss_fraction

def time_at_which_energy_loss_exceeds(times, tau, threshold, t_max):
    """
    Determine the time at which the cumulative energy loss fraction exceeds a
    given threshold.
    
    Parameters:
    - times: List or array of particle loss times from firm3d's gpu_tracing.
    - tau: Slowing-down time constant.
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

def alpha_loss_objective_from_times(times, tau, threshold, t_max, epsilon=1.0/25000, verbose=True):
    """
    Objective function based on the energy loss as a function of time.

    If threshold is >= 1, simply return the energy loss fraction.

    If threshold < 1, use the more sophisticated objective derived in the note
    20251112-01 Objective function for alpha particle loss.lyx

    The default value for epsilon corresponds to losing one particle out of 25000.

    Parameters:
    - times: List or array of particle loss times from firm3d's gpu_tracing.
    - tau: Slowing-down time constant.
    - threshold: The energy loss fraction threshold to exceed (between 0 and 1).
    - t_max: This value is returned if the loss fraction never exceeds the threshold.
    - epsilon: Small value to avoid divergence of log if losses are exactly 0.
    - verbose: If True, print detailed information.
    """
    number_loss_fraction, energy_loss_fraction = compute_energy_loss_fraction(times, tau)

    if threshold >= 1.0:
        # Simple objective based on energy loss fraction only.
        objective = -np.log10(energy_loss_fraction)
        case = 0
        if verbose:
            print(f"Number loss fraction: {number_loss_fraction}, energy loss fraction: {energy_loss_fraction}, alpha loss objective: {objective}")

    else:
        # If we made it here, then we are using the more sophisticated objective.
        t_exceed = time_at_which_energy_loss_exceeds(times, tau, threshold, t_max)
        if t_exceed < t_max:
            objective = -np.log10(t_exceed)
            case = 1
            if verbose:
                print(f"Alpha loss objective case 1: time to exceed threshold: {t_exceed}, objective: {objective}")
        else:
            objective = (
                np.log10(energy_loss_fraction + epsilon)
                -np.log10(t_max)
                -np.log10(threshold + epsilon)
            )
            case = 2
            if verbose:
                print(f"Alpha loss objective case 2: energy loss fraction: {energy_loss_fraction}, objective: {objective}")

    return objective, case
