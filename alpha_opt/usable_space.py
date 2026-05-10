import os
import time
import numpy as np
from vmecpp.simsopt_compat import Vmec

from . import DATA_DIR
from .pca import SurfaceWeightedPCA
from .surface import SurfaceGarabedianQuantiles
from .objective import get_objective
from .constants import ARIES_CS_MINOR_RADIUS

# Check if mpi4py is available
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mpi_available = True
except ImportError:
    rank = 0
    size = 1
    mpi_available = False


def measure_usable_space(
    surface_type="PCA",
    which_nfp="allNfp",
    n_pca_components=20,
    mpol=3,
    vmec_input="finite beta",
    nfp=4,
    aspect_ratio=6,
    min_for_each_dof=0.0,
    minutes=0.3,
    iota_threshold=0.0,
    print_every=10,
    max_B_target=12.0,
    max_B_iterations=1,
    save_intermediate=True,
    save_interval_seconds=30,
    save_dir=".",
):
    """Measure the fraction of parameter space in which vmec converges, using a
    weighted PCA or Garabedian quantile surface with max_B iteration machinery.

    Parameters sampled uniformly in [min_for_each_dof, 1 - min_for_each_dof].

    If mpi4py is installed, this function will use MPI to run multiple
    processes in parallel.

    Parameters
    ----------
    surface_type : str
        Either "PCA" or "Garabedian" to choose the surface parameterization.
        "PCA" uses weighted PCA with weighted quantile transform.
        "Garabedian" uses Garabedian basis with weighted quantile transform.
    which_nfp : str
        Either "allNfp" or "nfpAtLeast3" to choose the training dataset.
    n_pca_components : int
        Number of principal components to use (only for PCA surface type).
        Default is 20.
    mpol : int
        Poloidal mode number (ntor will be set equal to mpol for Garabedian 
        surface type). Default is 3.
    vmec_input : str
        Either "vacuum" or "finite beta" to use bundled VMEC input files,
        or a custom path to a VMEC input file. Default is "finite beta".
    nfp : int
        Number of field periods. Default is 4.
    aspect_ratio : float
        Aspect ratio of the surface. Default is 6.
    min_for_each_dof : float
        Minimum absolute value for each degree of freedom when sampling uniformly
        from [min_for_each_dof, 1 - min_for_each_dof]. Default is 0.1.
    minutes : float
        Number of minutes to run the test. Default is 0.3.
    iota_threshold : float
        Threshold for considering iota to be "good". Default is 0.2.
    print_every : int
        Print status every this many trials. Default is 10.
    max_B_target : float
        Target maximum magnetic field strength in Tesla. Default is 12.0.
    max_B_iterations : int
        Number of iterations to adjust phiedge to match max_B_target. Default is 1.
    save_intermediate : bool
        If True, each MPI rank periodically writes a plaintext checkpoint file
        with its running totals so results can be recovered if the job is killed.
        Default is True.
    save_interval_seconds : float
        How often (in seconds) each rank writes its checkpoint file.
        Default is 30.
    save_dir : str
        Directory in which to write checkpoint files. Default is "." (current
        working directory).

    Returns
    -------
    n_trials : int
        Total number of trials performed.
    n_successes : int
        Total number of successful VMEC runs.
    n_good_iota : int
        Total number of successful VMEC runs with good iota.
    success_fraction : float
        Fraction of trials that were successful.
    good_iota_fraction : float
        Fraction of trials with good iota.
    """
    minor_radius = ARIES_CS_MINOR_RADIUS
    major_radius = minor_radius * aspect_ratio

    # Validate inputs
    if surface_type not in ("PCA", "Garabedian"):
        raise ValueError(f"surface_type must be 'PCA' or 'Garabedian', got {surface_type}")
    if which_nfp not in ("allNfp", "nfpAtLeast3"):
        raise ValueError(f"which_nfp must be 'allNfp' or 'nfpAtLeast3', got {which_nfp}")

    # Set different random seed for each MPI process
    np.random.seed(rank)

    start_time = time.time()

    # Determine h5 file path based on surface_type and which_nfp
    if surface_type == "PCA":
        if which_nfp == "allNfp":
            h5_filename = "20260401-01_prepare_weighted_data_allNfp_PCA.h5"
        else:  # nfpAtLeast3
            h5_filename = "20260402-01_prepare_weighted_data_nfpAtLeast3_PCA.h5"
    else:  # Garabedian
        if which_nfp == "allNfp":
            h5_filename = "20260401-01_prepare_weighted_data_allNfp_Garabedian.h5"
        else:  # nfpAtLeast3
            h5_filename = "20260402-01_prepare_weighted_data_nfpAtLeast3_Garabedian.h5"

    h5_filepath = os.path.join(DATA_DIR, h5_filename)

    if vmec_input == "vacuum":
        vmec_input = os.path.join(DATA_DIR, "input.vmec")
    elif vmec_input == "finite beta":
        vmec_input = os.path.join(DATA_DIR, "input.finite_beta")

    # Create surface
    if surface_type == "PCA":
        surface = SurfaceWeightedPCA(
            nfp,
            major_radius,
            minor_radius,
            n_pca_components,
            filename=h5_filepath,
            exact_radii=True,
        )
        n_dofs = n_pca_components
    else:  # Garabedian
        surface = SurfaceGarabedianQuantiles(
            nfp=nfp,
            major_radius=major_radius,
            minor_radius=minor_radius,
            mpol=mpol,
            ntor=mpol,
            filename=h5_filepath,
            seed=rank,
            exact_radii=True,
        )
        n_dofs = len(surface.x)

    # Set up VMEC
    vmec = Vmec(vmec_input, verbose=False)
    vmec.indata.nfp = nfp

    # Set phiedge
    avg_B_estimate = max_B_target / np.sqrt(2)
    phiedge_estimate = np.pi * avg_B_estimate * minor_radius**2
    phiedge_high = phiedge_estimate * 2
    vmec.set("phiedge", phiedge_high)

    # Create a dummy objective function (we only care about VMEC convergence)
    def dummy_objective():
        return 0.0

    # Create wrapped objective that handles max_B iteration
    x_scale = np.ones(n_dofs)
    wrapped_objective = get_objective(
        vmec,
        surface,
        x_scale,
        dummy_objective,
        fail_val=1e10,
        max_B=max_B_target,
        max_B_iterations=max_B_iterations,
        phiedge=phiedge_high,
    )

    n_trials = 0
    n_successes = 0
    n_good_iota = 0

    checkpoint_path = os.path.join(save_dir, f"usable_space_rank{rank:04d}.txt")
    last_save_time = start_time

    def write_checkpoint():
        with open(checkpoint_path, "w") as _f:
            _f.write(f"rank={rank}\n")
            _f.write(f"n_trials={n_trials}\n")
            _f.write(f"n_successes={n_successes}\n")
            _f.write(f"n_good_iota={n_good_iota}\n")
            _f.write(f"elapsed_seconds={time.time() - start_time:.1f}\n")

    def print_status():
        success_frac = n_successes / n_trials if n_trials > 0 else 0
        print(
            f"[rank {rank:04d}] n_trials: {n_trials}  n_successes: {n_successes}  fraction: {success_frac:.4f}"
        )

    while True:
        elapsed_minutes = (time.time() - start_time) / 60
        if elapsed_minutes > minutes and n_trials > 0:
            # If the PCA takes a long time, make sure to complete at least one trial before breaking, so that we have some data to return.
            break

        # Sample parameters uniformly from [min_for_each_dof, 1 - min_for_each_dof]
        x = np.random.uniform(min_for_each_dof, 1.0 - min_for_each_dof, n_dofs)

        try:
            result = wrapped_objective(x)

            # Check if VMEC converged (result will be fail_val if VMEC failed)
            if result < 1e9:
                n_successes += 1
                iota = abs(vmec.wout.iotaf[-1])
                if iota > iota_threshold:
                    n_good_iota += 1
        except Exception:
            # VMEC failed to converge or other exception
            pass

        n_trials += 1

        if n_trials % print_every == 0:
            print_status()

        if save_intermediate:
            now = time.time()
            if now - last_save_time >= save_interval_seconds:
                write_checkpoint()
                last_save_time = now

    print_status()
    if save_intermediate:
        write_checkpoint()

    # Sum results across all MPI processes
    line_width = 60
    print("\n" + "=" * line_width)

    if mpi_available:
        # Use allreduce to compute sums and make them available on all ranks
        n_trials = comm.allreduce(n_trials, op=MPI.SUM)
        n_successes = comm.allreduce(n_successes, op=MPI.SUM)
        n_good_iota = comm.allreduce(n_good_iota, op=MPI.SUM)

        if rank == 0:
            print(f"FINAL SUMMARY (totals for {size} MPI processes)")
    else:
        print("FINAL SUMMARY (no MPI)")

    success_ratio = n_successes / n_trials if n_trials > 0 else 0
    good_iota_ratio = n_good_iota / n_trials if n_trials > 0 else 0

    if rank == 0:
        print("=" * line_width)
        print(f"Surface type: {surface_type}")
        print(f"Dataset: {which_nfp}")
        if surface_type == "PCA":
            print(f"Number of PCA components: {n_pca_components}")
        else:
            print(f"Poloidal/toroidal modes: mpol=ntor={mpol}")
        print(f"Total trials: {n_trials}")
        print(f"Total successes: {n_successes}")
        print(f"Total good iota (> {iota_threshold}): {n_good_iota}")
        print(f"Success fraction: {success_ratio:.6f}")
        print(f"Good iota fraction: {good_iota_ratio:.6f}")
        print("=" * line_width)

    return n_trials, n_successes, n_good_iota, success_ratio, good_iota_ratio
