import os
import numpy as np
# from vmecpp.simsopt_compat import Vmec
from simsopt.mhd.vmec import Vmec
from alpha_opt import DATA_DIR, get_DMerc_normalized, get_worst_DMerc_normalized

def test_DMerc_normalized_independent_of_B_and_size():
    """DMerc_normalized should be independent of the B field strength and size of the device."""
    vmec1 = Vmec(os.path.join(DATA_DIR, "wout_circular_model_tokamak_finiteBeta_output.nc"))
    vmec2 = Vmec(os.path.join(DATA_DIR, "wout_circular_model_tokamak_finiteBeta_2xB_output.nc"))
    vmec3 = Vmec(os.path.join(DATA_DIR, "wout_circular_model_tokamak_finiteBeta_2xSize_output.nc"))
    DMerc1 = get_DMerc_normalized(vmec1)
    DMerc2 = get_DMerc_normalized(vmec2)
    DMerc3 = get_DMerc_normalized(vmec3)

    # import matplotlib.pyplot as plt
    # plt.plot(vmec1.s_full_grid, DMerc1, label="Original")
    # plt.plot(vmec2.s_full_grid, DMerc2, label="2xB")
    # plt.plot(vmec3.s_full_grid, DMerc3, label="2xSize")
    # plt.xlabel("s")
    # plt.ylabel("DMerc_normalized")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    np.testing.assert_allclose(DMerc1, DMerc2, rtol=1e-12)
    np.testing.assert_allclose(DMerc1, DMerc3, rtol=0.016)

def test_DMerc_normalized_matches_desc_utils():
    """Compare to the desc_utils implementation for the same equilibrium."""
    vmec = Vmec(os.path.join(DATA_DIR, "wout_circular_model_tokamak_finiteBeta_output.nc"))
    DMerc_vmec = get_DMerc_normalized(vmec)[1:]  # Drop point at s=0

    # Uncomment these next lines to generate the reference values from desc_utils:

    # import desc.io
    # from desc.grid import LinearGrid
    # from desc_utils import Mercier_normalization
    # eq = desc.io.load("/Users/mattland/desc_utils/tests/inputs/circular_model_tokamak_finiteBeta_output.h5")
    # eq = eq[-1]
    # rho = np.sqrt(vmec.s_full_grid)[1:]  # Drop point at s=0
    # grid = LinearGrid(rho=rho, M=eq.M * 2, N=eq.N * 2, NFP=eq.NFP)
    # data = eq.compute(["D_Mercier", "V", "G", "p_r", "rho", "<|B|>_rms"], grid=grid)
    # normalization_desc = Mercier_normalization(data, eq.Psi)
    # DMerc_desc = data["D_Mercier"] / normalization_desc
    # DMerc_desc = grid.compress(DMerc_desc)
    # np.set_printoptions(linewidth=500)
    # print(DMerc_desc)

    # Data copied from stdout of the above code block.
    DMerc_desc = np.array([0.02822635, 0.03588178, 0.04353607, 0.05118882, 0.05883973, 0.06648853, 0.07413502, 0.08177902, 0.08942041, 0.09705909, 0.10469499, 0.11232808, 0.11995833, 0.12758574, 0.13521032, 0.1428321, 0.15045112, 0.15806741, 0.16568102, 0.17329203, 0.18090048, 0.18850644, 0.19610999, 0.20371119, 0.21131012, 0.21890685, 0.22650145, 0.234094, 0.24168458, 0.24927325, 0.2568601, 0.26444521, 0.27202863, 0.27961045, 0.28719075, 0.29476959, 0.30234704, 0.30992319, 0.31749809, 0.32507181, 0.33264442, 0.340216, 0.34778659, 0.35535627, 0.36292509, 0.37049312, 0.37806039, 0.38562698, 0.39319292, 0.40075826, 0.40832304, 0.4158873, 0.42345107, 0.43101438, 0.43857725, 0.44613969, 0.45370172, 0.46126334, 0.46882455, 0.47638533, 0.48394568, 0.49150555, 0.49906493, 0.50662378, 0.51418203, 0.52173963, 0.52929652, 0.53685261, 0.54440782, 0.55196206, 0.55951521, 0.56706716, 0.57461779, 0.58216695, 0.58971451, 0.59726029, 0.60480414, 0.61234588, 0.61988533, 0.62742227, 0.63495652, 0.64248784, 0.65001603, 0.65754083, 0.66506201, 0.67257932, 0.68009249, 0.68760126, 0.69510536, 0.7026045, 0.71009839, 0.71758674, 0.72506925, 0.73254562, 0.74001555, 0.7474787, 0.75493479, 0.76238348, 0.76982447, 0.77725743, 0.78468205, 0.79209802, 0.79950502, 0.80690274, 0.81429088, 0.82166915, 0.82903724, 0.83639487, 0.84374178, 0.8510777, 0.85840237, 0.86571557, 0.87301706, 0.88030665, 0.88758416, 0.89484941, 0.90210227, 0.90934262, 0.91657037, 0.92378547, 0.9309879, 0.93817765, 0.94535479, 0.9525194, 0.9596716, 0.9668116, 0.97393961, 0.98105592, 0.98816088, 0.9952549, 1.00233844, 1.00941207, 1.01647639, 1.02353211, 1.03058002, 1.03762101, 1.04465604, 1.05168619, 1.05871265, 1.06573673, 1.07275984, 1.07978354, 1.0868095, 1.09383956, 1.10087567, 1.10791997, 1.11497475, 1.12204246, 1.12912574, 1.13622741, 1.14335047, 1.15049815, 1.15767385, 1.16488121, 1.17212408, 1.17940655, 1.18673293, 1.19410778, 1.20153591, 1.20902238, 1.21657251, 1.22419189, 1.23188636, 1.23966206, 1.24752536, 1.25548293, 1.26354172, 1.27170893, 1.27999203, 1.28839877, 1.29693713, 1.30561536, 1.31444191, 1.3234255, 1.332575, 1.34189948, 1.35140819, 1.36111046, 1.37101575, 1.38113355, 1.39147338, 1.40204468, 1.41285682, 1.42391897, 1.43524009, 1.44682876, 1.45869314, 1.47084085, 1.48327882, 1.49601315, 1.50904897, 1.52239021, 1.53603946, 1.54999762, 1.56426376, 1.5788347, 1.59370472, 1.60886516, 1.62430395, 1.6400051])

    # import matplotlib.pyplot as plt
    # plt.plot(vmec.s_full_grid[1:], DMerc_vmec, label="VMEC")
    # plt.plot(vmec.s_full_grid[1:], DMerc_desc, ':', label="DESC")
    # plt.xlabel("s")
    # plt.tight_layout()
    # plt.legend()
    # plt.show()

    np.testing.assert_allclose(DMerc_vmec, DMerc_desc, rtol=0.0002)


def test_worst_DMerc_normalized_full_range():
    """Test that get_worst_DMerc_normalized returns the minimum over the full range."""
    vmec = Vmec(os.path.join(DATA_DIR, "wout_circular_model_tokamak_finiteBeta_output.nc"))
    worst = get_worst_DMerc_normalized(vmec, s_min=0.0, s_max=1.0)
    
    # Should match the minimum of the full array
    DMerc_normalized = get_DMerc_normalized(vmec)
    expected = np.min(DMerc_normalized)
    
    np.testing.assert_allclose(worst, expected, rtol=1e-12)