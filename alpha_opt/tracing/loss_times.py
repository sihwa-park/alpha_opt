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

def variance_case2(times, tau, t_max, epsilon):
    """
    Closed-form (delta method) variance of the case-2 objective:
        obj = log10(ELF + eps) - log10(t_max) - log10(threshold + eps)
    
    Steps:
        1. Compute per-particle contributions X_i (exact)
        2. Compute Var(ELF) = s² / N  (exact, no approximation) where s² = (1/(N-1)) Σ(X_i - ELF)²  is the unbiased sample variance of the X_i (Bessel's correction, ddof=1)
        3. Propagate through log10 via delta method (first-order Taylor)
    """
    N = len(times)

    # step 1: per-particle energy-loss contributions
    # survivors (t_i == t_max) contribute 0; lost particles contribute exp(-t_i/tau)
    X = np.where(times < t_max, np.exp(-times / tau), 0.0)

    # step 2: ELF and its variance (exact, no Taylor here)
    ELF = X.mean()
    var_ELF = X.var(ddof=1) / N       # Var(sample mean) = sample variance / N

    # step 3: delta method through log10(ELF + eps)
    # h(u) = log10(u + eps),  h'(u) = 1 / ((u + eps) * ln10)
    ln10 = np.log(10.0)
    h_prime = 1.0 / ((ELF + epsilon) * ln10)
    var_obj = h_prime**2 * var_ELF    # Var(h(ELF)) ≈ h'(ELF)^2 * Var(ELF)

    return var_obj, ELF

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
    var_obj = 1e-5  # default value for case 0 and case 1
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
            var_obj, _ = variance_case2(times, tau, t_max, epsilon)
            case = 2
            if verbose:
                print(f"Alpha loss objective case 2: energy loss fraction: {energy_loss_fraction}, objective: {objective}")

    return objective, case, var_obj
