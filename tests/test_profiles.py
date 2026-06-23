import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import elementary_charge
import scipy.stats as stats
from scipy.interpolate import interp1d
import vmecpp
from alpha_opt import DATA_DIR
from alpha_opt.tracing.profiles import n_m3_func, T_keV_func, DT_reaction_rate, relative_DT_reaction_rate_for_our_profiles, sample_alpha_birth_s

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

    np.testing.assert_allclose(pressure_vmec, pressure_should_be, rtol=1e-11)

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


def test_sample_alpha_birth_s_distribution():
    """Test that samples follow the expected distribution using statistical tests."""
    # Generate a large sample
    n_samples = 10000
    np.random.seed(42)  # For reproducible tests
    samples = [sample_alpha_birth_s() for _ in range(n_samples)]
    
    # Test 1: Basic properties
    assert all(0 <= s <= 1 for s in samples), "All samples should be in [0,1]"
    
    # Test 2: Compare empirical CDF to theoretical CDF
    # First, compute theoretical CDF by numerical integration
    # Avoid s=1 where T_keV_func becomes zero
    s_grid = np.linspace(0, 0.99, 1000)
    pdf_values = []
    
    for s in s_grid:
        try:
            val = relative_DT_reaction_rate_for_our_profiles(s)
            if np.isfinite(val) and val >= 0:
                pdf_values.append(val)
            else:
                pdf_values.append(0.0)
        except:
            pdf_values.append(0.0)
    
    pdf_values = np.array(pdf_values)
    
    # Normalize to get proper PDF
    pdf_normalized = pdf_values / np.trapezoid(pdf_values, s_grid)
    cdf_theoretical = np.cumsum(pdf_normalized) * (s_grid[1] - s_grid[0])
    
    # Kolmogorov-Smirnov test
    # Create interpolated CDF function
    cdf_func = interp1d(s_grid, cdf_theoretical, bounds_error=False, fill_value=(0, 1))
    
    # Apply K-S test
    ks_statistic, p_value = stats.kstest(samples, cdf_func)
    assert p_value > 0.01, f"K-S test failed with p-value {p_value}"


def test_sample_alpha_birth_s_histogram():
    """Test distribution by comparing histograms."""
    n_samples = 50000
    np.random.seed(123)  # For reproducible tests
    samples = [sample_alpha_birth_s() for _ in range(n_samples)]
    
    # Filter samples to avoid the problematic region near s=1
    samples_filtered = [s for s in samples if s < 0.95]
    
    # Create histogram of filtered samples
    bins = np.linspace(0, 0.95, 40)
    hist_observed, _ = np.histogram(samples_filtered, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Compute expected histogram from theoretical distribution
    expected_pdf = []
    for s in bin_centers:
        try:
            val = relative_DT_reaction_rate_for_our_profiles(s)
            if np.isfinite(val) and val >= 0:
                expected_pdf.append(val)
            else:
                expected_pdf.append(0.0)
        except:
            expected_pdf.append(0.0)
    
    expected_pdf = np.array(expected_pdf)
    
    # Normalize
    normalization = np.trapezoid(expected_pdf, bin_centers)
    if normalization > 0:
        expected_pdf /= normalization
    
    # Compare only in regions where we have reasonable expected values
    mask = expected_pdf > np.max(expected_pdf) * 1e-3  # Focus on significant parts of distribution
    if np.any(mask):
        relative_error = np.abs(hist_observed[mask] - expected_pdf[mask]) / (expected_pdf[mask] + 1e-10)
        assert np.mean(relative_error) < 0.2, f"Histogram doesn't match expected distribution, mean relative error: {np.mean(relative_error)}"
    
    # Also check that the overall shape is reasonable by comparing first few moments
    sample_mean_filtered = np.mean(samples_filtered)
    theoretical_mean = np.trapezoid(bin_centers * expected_pdf, bin_centers)
    assert abs(sample_mean_filtered - theoretical_mean) / theoretical_mean < 0.1, "Sample mean differs significantly from theoretical mean"


def test_sample_alpha_birth_s_moments():
    """Test that sample moments match theoretical moments."""
    n_samples = 100000
    np.random.seed(456)  # For reproducible tests
    samples = np.array([sample_alpha_birth_s() for _ in range(n_samples)])
    
    # Compute theoretical moments by numerical integration
    # Avoid s=1 where T_keV_func becomes zero
    s_grid = np.linspace(0, 0.99, 10000)
    pdf_values = []
    
    for s in s_grid:
        try:
            val = relative_DT_reaction_rate_for_our_profiles(s)
            if np.isfinite(val) and val >= 0:
                pdf_values.append(val)
            else:
                pdf_values.append(0.0)
        except:
            pdf_values.append(0.0)
    
    pdf_values = np.array(pdf_values)
    pdf_normalized = pdf_values / np.trapezoid(pdf_values, s_grid)
    
    # Theoretical moments
    theoretical_mean = np.trapezoid(s_grid * pdf_normalized, s_grid)
    theoretical_var = np.trapezoid((s_grid - theoretical_mean)**2 * pdf_normalized, s_grid)
    
    # Sample moments
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)
    
    # Test with reasonable tolerance (accounting for finite sample size)
    std_error_mean = np.sqrt(sample_var / n_samples)
    std_error_var = np.sqrt(2 * sample_var**2 / n_samples)  # approximate
    
    assert abs(sample_mean - theoretical_mean) < 3 * std_error_mean, \
        f"Sample mean {sample_mean} differs from theoretical {theoretical_mean} by more than 3 std errors"
    assert abs(sample_var - theoretical_var) < 3 * std_error_var, \
        f"Sample variance {sample_var} differs from theoretical {theoretical_var} by more than 3 std errors"


def test_sample_alpha_birth_s_bounds_and_properties():
    """Test basic properties and edge cases."""
    # Test output bounds
    np.random.seed(789)  # For reproducible tests
    for _ in range(1000):
        s = sample_alpha_birth_s()
        assert 0 <= s <= 1, f"Sample {s} outside [0,1] bounds"
    
    # Test reproducibility with fixed seed
    np.random.seed(42)
    samples1 = [sample_alpha_birth_s() for _ in range(100)]
    
    np.random.seed(42)
    samples2 = [sample_alpha_birth_s() for _ in range(100)]
    
    assert samples1 == samples2, "Function should be reproducible with same seed"
    
    # Test that function terminates quickly (no infinite loops)
    # We'll use a simple timeout by checking that many samples can be generated quickly
    import time
    start_time = time.time()
    np.random.seed(999)
    test_samples = [sample_alpha_birth_s() for _ in range(100)]
    elapsed_time = time.time() - start_time
    
    assert elapsed_time < 5.0, f"Function took too long ({elapsed_time:.2f}s), possible infinite loop"
    assert len(test_samples) == 100, "Not all samples were generated"