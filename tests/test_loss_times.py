import os
import pandas as pd
import numpy as np

from alpha_opt import (
    DATA_DIR,
    compute_energy_loss_fraction,
    time_at_which_energy_loss_exceeds,
    alpha_loss_objective_from_times,
)


def test_compute_energy_loss_fraction():
    # Load test data
    data_path = os.path.join(DATA_DIR, "particle_data.csv")
    df = pd.read_csv(data_path)
    times = df["last_time"].values

    # For this large tau, the particle and energy loss fractions should be nearly equal
    tau = 0.1
    number_loss_fraction, energy_loss_fraction = compute_energy_loss_fraction(
        times, tau
    )
    np.testing.assert_allclose(number_loss_fraction, 0.191, rtol=1e-5)
    np.testing.assert_allclose(energy_loss_fraction, 0.190227, rtol=1e-5)

    # Try a smaller tau, which should reduce the energy loss fraction
    tau = 1e-4
    number_loss_fraction, energy_loss_fraction = compute_energy_loss_fraction(
        times, tau
    )
    np.testing.assert_allclose(number_loss_fraction, 0.191, rtol=1e-5)
    np.testing.assert_allclose(energy_loss_fraction, 0.058665, rtol=1e-5)


def test_time_at_which_energy_loss_exceeds():
    # Load test data
    data_path = os.path.join(DATA_DIR, "particle_data.csv")
    df = pd.read_csv(data_path)
    times = df["last_time"].values

    tau = 0.1
    threshold = 0.1
    t_max = 999.9
    t_exceed = time_at_which_energy_loss_exceeds(times, tau, threshold, t_max)
    expected_t_exceed = 1.247874e-04  # Using 25,000 particles instead of 1000
    np.testing.assert_allclose(t_exceed, expected_t_exceed, rtol=0.005)

    # As long as tau is >> t_exceed, the result should be insensitive to tau
    tau_large = 10.0
    t_exceed_large_tau = time_at_which_energy_loss_exceeds(
        times, tau_large, threshold, t_max
    )
    np.testing.assert_allclose(t_exceed_large_tau, expected_t_exceed, rtol=0.005)

    # If tau is comparable to t_exceed, the time to reach the threshold should be larger
    tau_small = 3e-4
    t_exceed_small_tau = time_at_which_energy_loss_exceeds(
        times, tau_small, threshold, t_max
    )
    print("\nt_exceed with original tau:", t_exceed)
    print("t_exceed with small tau:   ", t_exceed_small_tau)
    np.testing.assert_array_less(t_exceed, t_exceed_small_tau)

    # Check the case in which the loss never exceeds the threshold
    high_threshold = 0.99
    t_exceed_high = time_at_which_energy_loss_exceeds(times, tau, high_threshold, t_max)
    assert t_exceed_high == t_max


def test_alpha_loss_objective_from_times():
    # Load test data
    data_path = os.path.join(DATA_DIR, "particle_data.csv")
    df = pd.read_csv(data_path)
    times = df["last_time"].values

    tau = 0.15
    t_max = 0.2
    epsilon = 1.0 / 25000  # Corresponds to losing one particle out of 25,000

    # Test the case where threshold >= 1: Should use the simpler objective based on energy loss fraction only
    high_threshold = 10.0
    objective_value_high_threshold, which_case = alpha_loss_objective_from_times(
        times, tau, high_threshold, t_max, epsilon, verbose=False
    )
    number_loss_fraction, energy_loss_fraction = compute_energy_loss_fraction(
        times, tau
    )
    np.testing.assert_equal(which_case, 0)
    np.testing.assert_allclose(
        objective_value_high_threshold, -np.log10(energy_loss_fraction), rtol=1e-5
    )

    # Test the more sophisticated objective, case 1:
    # Time to exceed threshold is less than t_max
    threshold = 0.1
    objective_value, which_case = alpha_loss_objective_from_times(
        times, tau, threshold, t_max, epsilon, verbose=False
    )
    np.testing.assert_equal(which_case, 1)
    t_exceed = 1.248e-4
    expected_objective = -np.log10(t_exceed)
    np.testing.assert_allclose(objective_value, expected_objective, rtol=0.0005)

    # Test the more sophisticated objective, case 2:
    # Losses do not exceed threshold in t_max
    threshold = 0.3
    short_t_max = 3e-3
    objective_value, which_case = alpha_loss_objective_from_times(
        times, tau, threshold, short_t_max, epsilon, verbose=False
    )
    np.testing.assert_equal(which_case, 2)
    # Objective should be less than -log10(0.003) = 2.522
    expected_objective = 2.325648
    np.testing.assert_allclose(objective_value, expected_objective, rtol=0.0005)

    # Now check continuity of the objective near the boundary between cases 1
    # and 2
    threshold_boundary = 0.191197
    threshold = threshold_boundary * 0.99  # Slightly below boundary
    objective_value_1, which_case = alpha_loss_objective_from_times(
        times, tau, threshold, short_t_max, epsilon, verbose=False
    )
    np.testing.assert_equal(which_case, 1)
    threshold = threshold_boundary * 1.01  # Slightly above boundary
    objective_value_2, which_case = alpha_loss_objective_from_times(
        times, tau, threshold, short_t_max, epsilon, verbose=False
    )
    np.testing.assert_equal(which_case, 2)
    expected_objective = -np.log10(short_t_max)
    np.testing.assert_allclose(objective_value_1, expected_objective, rtol=0.01)
    np.testing.assert_allclose(objective_value_2, expected_objective, rtol=0.01)
    