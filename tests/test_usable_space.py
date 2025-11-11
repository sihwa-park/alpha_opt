import numpy as np
from alpha_opt.usable_space import measure_usable_space_pca


def test_usable_space_pca():
    n_trials, n_successes, n_good_iota, success_fraction, n_good_iota_fraction = (
        measure_usable_space_pca(
            minutes=0.1,
        )
    )
    np.testing.assert_array_less(0, n_trials)
    np.testing.assert_array_less(0, n_successes)
    np.testing.assert_array_less(success_fraction, 1.0)
    np.testing.assert_array_less(0.0, success_fraction)
    np.testing.assert_array_less(0, n_good_iota)
    np.testing.assert_array_less(n_good_iota, n_trials)
    np.testing.assert_array_less(n_good_iota_fraction, 1.0)
    np.testing.assert_array_less(0.0, n_good_iota_fraction)
