import os
import pandas as pd
import numpy as np

from alpha_opt import DATA_DIR, time_at_which_energy_loss_exceeds

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
    t_exceed_large_tau = time_at_which_energy_loss_exceeds(times, tau_large, threshold, t_max)
    np.testing.assert_allclose(t_exceed_large_tau, expected_t_exceed, rtol=0.005)

    # If tau is comparable to t_exceed, the time to reach the threshold should be larger
    tau_small = 3e-4
    t_exceed_small_tau = time_at_which_energy_loss_exceeds(times, tau_small, threshold, t_max)
    print("\nt_exceed with original tau:", t_exceed)
    print("t_exceed with small tau:   ", t_exceed_small_tau)
    np.testing.assert_array_less(t_exceed, t_exceed_small_tau)

    # Check the case in which the loss never exceeds the threshold
    high_threshold = 0.99
    t_exceed_high = time_at_which_energy_loss_exceeds(times, tau, high_threshold, t_max)
    assert t_exceed_high == t_max