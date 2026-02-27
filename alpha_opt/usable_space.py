import os
import time
import numpy as np
from sklearn.preprocessing import QuantileTransformer, RobustScaler, FunctionTransformer
from simsopt._core import ObjectiveFailure
from vmecpp.simsopt_compat import Vmec

from . import DATA_DIR
from .pca import SurfacePCAGarabedian, SurfacePCARealSpace
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


def measure_usable_space_pca(
    nfp=4,
    aspect_ratio=6,
    n_dimensions=7,
    x_max=2,
    minutes=0.3,
    iota_threshold=0.2,
    print_every=10,
    vmec_input="vacuum",
    transform1=QuantileTransformer(),
    transform2=RobustScaler(),
    space="Garabedian",
):
    """Measure the fraction of parameter space in which vmec converges, using a
    PCA surface.

    Each degree of freedom is sampled uniformly in [-x_max, x_max].

    If mpi4py is installed, this function will use MPI to run multiple
    processes in parallel.

    You can set transform1=None to skip the first transformation.

    You can set vmec_input="vacuum" or "finite beta" to use the default vacuum
    and finite-beta vmec input files included with alpha_opt.

    The argument `space` can be either "Garabedian" or "RealSpace" to choose
    between SurfacePCAGarabedian and SurfacePCARealSpace.

    Parameters
    ----------
    nfp : int
        Number of field periods.
    aspect_ratio : float
        Aspect ratio of the surface.
    n_dimensions : int
        Number of PCA dimensions to use.
    x_max : float
        Maximum absolute value of each PCA parameter.
    minutes : float
        Number of minutes to run the test.
    iota_threshold : float
        Threshold for considering iota to be "good".
    print_every : int
        Print status every this many trials.
    transform1 : sklearn transformer
        First transformer, to apply before PCA.
    transform2 : sklearn transformer
        Second transformer, to apply after PCA.
    space : str
        The type of PCA space to use ("Garabedian" or "RealSpace").

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

    if transform1 is None:
        transform1 = FunctionTransformer()

    # Set different random seed for each MPI process
    np.random.seed(rank)

    start_time = time.time()

    if space == "Garabedian":
        surf = SurfacePCAGarabedian(
            nfp,
            major_radius,
            minor_radius,
            n_dimensions,
            transform1=transform1,
            transform2=transform2,
        )
    elif space == "RealSpace":
        surf = SurfacePCARealSpace(
            nfp,
            major_radius,
            minor_radius,
            n_dimensions,
            transform1=transform1,
            transform2=transform2,
        )
    else:
        raise ValueError(f"Unknown space type: {space}")
    
    if vmec_input == "vacuum":
        vmec_input = os.path.join(DATA_DIR, "input.vmec")
    elif vmec_input == "finite beta":
        vmec_input = os.path.join(DATA_DIR, "input.finite_beta")

    vmec = Vmec(vmec_input, verbose=False)
    vmec.boundary = surf
    vmec.indata.nfp = (
        nfp  # Vmec++ does not automatically get nfp from the boundary surface!
    )

    n_trials = 0
    n_successes = 0
    n_good_iota = 0

    def print_status():
        print(
            f"n_trials: {n_trials}  n_successes: {n_successes}  fraction: {n_successes / n_trials}"
        )

    while True:
        elapsed_minutes = (time.time() - start_time) / 60
        if elapsed_minutes > minutes:
            break

        surf.x = (np.random.rand(n_dimensions) * 2 - 1) * x_max
        try:
            vmec.run()
            # Temporary workaround until vmecpp bug is fixed (vmecpp replaces
            # the boundary object when run)
            vmec.boundary = surf
            n_successes += 1
            iota = abs(vmec.wout.iotaf[-1])
            # print(f"iota: {iota}")
            if iota > iota_threshold:
                n_good_iota += 1
        except ObjectiveFailure:
            pass

        n_trials += 1

        if n_trials % print_every == 0:
            print_status()

    print_status()

    # Sum results across all MPI processes
    line_width = 60
    print("\n" + "=" * line_width)

    if mpi_available:
        # Use allreduce to compute sums and make them available on all ranks in a single collective
        n_trials = comm.allreduce(n_trials, op=MPI.SUM)
        n_successes = comm.allreduce(n_successes, op=MPI.SUM)
        n_good_iota = comm.allreduce(n_good_iota, op=MPI.SUM)

        if rank == 0:
            print(f"FINAL SUMMARY (totals for {size} MPI processes)")
    else:
        print("FINAL SUMMARY (no MPI)")

    success_ratio = n_successes / n_trials
    good_iota_ratio = n_good_iota / n_trials
    if rank == 0:
        print("=" * line_width)
        print(f"Total trials: {n_trials}")
        print(f"Total successes: {n_successes}")
        print(f"Total good iota (> {iota_threshold}): {n_good_iota}")
        print(f"Success fraction: {success_ratio:.6f}")
        print(f"Good iota fraction: {good_iota_ratio:.6f}")
        print("=" * line_width)

    return n_trials, n_successes, n_good_iota, success_ratio, good_iota_ratio
