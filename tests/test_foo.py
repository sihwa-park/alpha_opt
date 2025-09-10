import numpy as np

from alpha_opt import foo

def test_foo():
    np.testing.assert_equal(foo(7), 8)
    np.testing.assert_equal(foo(3.2), 4.2)
    x = np.array([1, 2, 3.3])
    np.testing.assert_allclose(foo(x), np.array([2, 3, 4.3]))
