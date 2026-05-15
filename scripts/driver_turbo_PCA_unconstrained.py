#!/usr/bin/env python

# This script borrows heavily from the BoTorch TuRBO example at
# https://botorch.org/docs/tutorials/turbo_1/

from dataclasses import dataclass
import os
import pickle
import json
import time
import traceback
import math
import numpy as np
import pandas as pd
import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from vmecpp.simsopt_compat import Vmec
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine
import warnings

from alpha_opt import SurfaceWeightedPCA, get_worst_DMerc_normalized
from alpha_opt.objective import get_objective
from alpha_opt.gpu_tracing import compute_alpha_loss
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
if rank == 0:
    os.environ["OMP_NUM_THREADS"] = "12"   # CPU-only master
else:
    os.environ["OMP_NUM_THREADS"] = "13"   # GPU workers

aspect_ratio = 6.0
#major_radius = 1.0
#minor_radius = major_radius / aspect_ratio
# minor_radius = 1.70442622782386  # ARIES-CS
minor_radius = 3.1 / (aspect_ratio**0.38)
major_radius = minor_radius * aspect_ratio

max_B_target = 12.0
max_B_iterations = 1

vacuum = False
# vacuum = True

n_particles = 25000
t_max = 1e-1
tau = 0.1

# Trace only until losses exceed this fraction:
maxloss = 0.02
t_block = 1e-3

# See 20251109-02 GPU tracing - convergence with respect to tol and min_dt.docx
tol = 1e-6
min_dt = 1e-9

pca_data_file = "/pscratch/sd/l/landrema/20260402-01-alpha_optimization_unconstrained_parameter_scans/20260402-01_prepare_weighted_data_nfpAtLeast3_PCA.h5"

# Dimensions of the parameter space to optimize in: (number of principal components)
dim_x = 20

# How to handle cases in which VMEC does not converge:
# If False, Ax will classify the run as FAILED.
# If True, Ax will classify the run as COMPLETED with a large value of the objective.
treat_failures_as_big_number = True
fail_val = 5.5  # Like losing maxloss of the energy at 10^{-5.5} sec.
DMerc_fail_val = -0.5

# Max number of evaluations to perform
num_evals = 10000

# Number of trials to generate at once in batches
batch_size = 1

max_wallclock_minutes = 60 * 3 - 5  # Leave 5 minutes buffer at end of job

specify_zero_variance = False

state_file = "ax_client_state.json"
csv_file = "evals.csv"
# Save state every time this many function evals are completed:
save_frequency = 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
proc0 = (rank == 0)

###############################################################################
# Timing / logging utilities
###############################################################################
# Record start time for elapsed time tracking (must be before any logging)
# Broadcast from rank 0 so all processes have exactly the same start_time
if proc0:
    start_time = time.time()
else:
    start_time = None
start_time = comm.bcast(start_time, root=0)


def _format_elapsed(elapsed: float) -> str:
    """Return elapsed seconds as zero-padded string with 6 digits before decimal
    and 3 after (thousandths precision)."""
    # thousandths = int(round(elapsed * 1000))
    # secs = thousandths // 1000
    # frac = thousandths % 1000
    # return f"{secs:06d}.{frac:03d}"
    return f"{elapsed:010.3f}"

def log(*parts, sep=" ", end="\n"):
    """Unified logging function.

    Produces lines beginning with "[elapsed, rank] " where elapsed has the form
    000123.457 (six digits before decimal, three after) and rank is zero-padded to 4.
    Modify formatting here to affect all log messages project-wide.
    """
    elapsed = time.time() - start_time
    prefix = f"[{_format_elapsed(elapsed)} {rank:04d}] "
    print(prefix + sep.join(str(p) for p in parts), end=end, flush=True)

if proc0:
    log(f"Running with {size} MPI ranks.")

np.set_printoptions(linewidth=300)

vmec = Vmec("input.vmec", verbose=(rank == 1))
avg_B_estimate = max_B_target / np.sqrt(2)
phiedge_estimate = np.pi * avg_B_estimate * minor_radius**2
# Initially use a phiedge which should have Bmax >> 12 T
phiedge_high = phiedge_estimate * 2
vmec.set("phiedge", phiedge_high)

surface = SurfaceWeightedPCA(
    vmec.indata.nfp,
    major_radius,
    minor_radius,
    dim_x,
    filename=pca_data_file,
)
x_scale = np.ones(dim_x)

#qs = QuasisymmetryRatioResidual(
#    vmec, 
#    np.linspace(0, 1, 11),
#    helicity_m=1,
#    helicity_n=1,
#)
#
#wrapped_objective = get_objective(vmec, surface, x_scale, qs.total, fail_val=fail_val)

def raw_objective():
    wout_filename = "wout_tmp.nc"
    wout_save_start_time = time.time()
    vmec.wout.save(wout_filename)
    print("Time to save wout:", time.time() - wout_save_start_time)
    objective = compute_alpha_loss(
        wout_filename, n_particles=n_particles, t_max=t_max, tau=tau, min_dt=min_dt,
        maxloss=maxloss, t_block=t_block,
        tol=tol,
        vacuum=vacuum,
    )
    return objective

wrapped_objective = get_objective(
    vmec,
    surface,
    x_scale,
    raw_objective,
    fail_val=fail_val,
    max_B=max_B_target,
    max_B_iterations=max_B_iterations,
    phiedge=phiedge_high,
)

# Record start time for elapsed time tracking
start_time = time.time()

# Define parameter space
param_names = [f"PC{i}" for i in range(dim_x)]
dim = dim_x

# Bounds are [0, 1] for each PCA component
lb = np.zeros(dim, dtype=np.float64)
ub = np.ones(dim, dtype=np.float64)
bounds = torch.stack([torch.from_numpy(lb), torch.from_numpy(ub)])

if proc0:
    log(f"Lower bounds: {lb}")
    log(f"Upper bounds: {ub}")

# MPI tags for different message types
WORK_TAG = 1
RESULT_TAG = 2
STOP_TAG = 3

# Device and dtype for PyTorch tensors
device = torch.device("cpu")  # Keep on CPU for reproducibility
dtype = torch.double

# Suppress numerical warnings
warnings.filterwarnings(
    "ignore",
    message=r"Very small noise values detected\. This will likely lead to numerical instabilities\. Rounding small noise values up to 1e-06\."
)

###############################################################################
# TuRBO State and Functions
###############################################################################

@dataclass
class TurboState:
    """Turbo state used to track the recent history of the trust region."""
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = float("inf")  # For minimization
    restart_triggered: bool = False

    def __post_init__(self):
        """Post-initialize the state of the trust region."""
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_turbo_state(state: TurboState, Y_next: np.ndarray) -> TurboState:
    """Update the state of the trust region based on the new function values (minimization)."""
    Y_min = float(np.min(Y_next))
    if Y_min < state.best_value - 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = min(state.best_value, Y_min)
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def normalize_point(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Normalize point from original space to [0, 1]."""
    return (x - lb) / (ub - lb)


def unnormalize_point(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Unnormalize point from [0, 1] to original space."""
    return lb + x * (ub - lb)


def generate_turbo_batch(
    state: TurboState,
    model: SingleTaskGP,
    X_turbo: torch.Tensor,  # Evaluated points in [0, 1]^d
    Y_turbo: torch.Tensor,  # Function values (standardized)
    batch_size: int,
    n_candidates: int = None,  # Number of candidates for Thompson sampling
    num_restarts: int = 10,
    raw_samples: int = 512,
    acqf: str = "ts",  # "ei" or "ts"
) -> torch.Tensor:
    """Generate a new batch of points using TuRBO."""
    assert acqf in ("ts", "ei")
    assert X_turbo.min() >= 0.0
    assert X_turbo.max() <= 1.0
    assert torch.all(torch.isfinite(Y_turbo))
    
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X_turbo.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X_turbo[Y_turbo.argmin(), :].clone()  # Use minimum for minimization
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        d = X_turbo.shape[-1]
        sobol = SobolEngine(d, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / d, 1.0)
        mask = torch.rand(n_candidates, d, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, d - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, d).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, Y_turbo.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next


def fit_gp_model(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    """Fit a GP model to the observed data."""
    # Standardize Y
    Y_train = (Y - Y.mean()) / Y.std()
    
    # Create covariance module
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=X.shape[-1], lengthscale_constraint=Interval(0.005, 4.0))
    )
    
    # Create and fit model
    model = SingleTaskGP(X, Y_train, covar_module=covar_module, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    # Always use Cholesky
    with gpytorch.settings.max_cholesky_size(float("inf")):
        fit_gpytorch_mll(mll)
    
    return model



def run_objective(params, trial_index):
    """Evaluate the objective function for given parameters"""
    run_objective_start_time = time.time()
    log(f"Evaluating trial {trial_index} with params: {list(params.values())}")
    
    # Create evaluation directory
    eval_dir = f"evals/eval{trial_index:06d}"
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save current working directory and change to eval directory
    original_cwd = os.getcwd()
    os.chdir(eval_dir)
    
    x = np.array([params[name] for name in param_names])
    objective_start_time = time.time()
    try:
        val = wrapped_objective(x)
        if val == fail_val:
            raise Exception("Objective failed")
        worst_dmerc = get_worst_DMerc_normalized(vmec, 0.2, 0.95)
        objective_time = time.time() - objective_start_time
        log(f"Objective evaluation for eval {trial_index} took {objective_time:.1f} seconds")
        log(f"  Loss: {val}, Worst DMerc: {worst_dmerc}")

        with open("results.txt", "a") as f:
            f.write(f"worst_DMerc = {worst_dmerc}\n")

        return_val = {"loss": val, "worst_dmerc": worst_dmerc}
    except Exception as e:
        objective_time = time.time() - objective_start_time
        log(f"Objective evaluation failed after {objective_time:.1f} seconds. {e}")
        log(f"Full traceback:\n{traceback.format_exc()}")
        return_val = {"loss": fail_val, "worst_dmerc": DMerc_fail_val}
    finally:
        # Always return to original directory
        os.chdir(original_cwd)

    log(f"Total time for run_objective for eval {trial_index}: {time.time() - run_objective_start_time:.1f} seconds")
    return return_val

def save_vmec_files(extension):
    if not proc0:
        return
    
    # Ensure wout and indata are updated with the state vector:
    # best_qs = qs.total()
    vmec.run()
    log(f"{extension} x:", surface.x)
    # log(f"{extension} qs.total():", best_qs)

    # Save results:
    vmec.wout.save(f"wout_{extension}.nc")
    # vmec.set_indata()
    with open(f"input.{extension}", "w") as f:
        f.write(vmec.indata.to_json(indent=2))


def save_turbo_state(X_turbo, Y_turbo, trial_status, worst_dmerc_by_trial=None):
    """Save TuRBO state to CSV file.
    
    Args:
        X_turbo: torch.Tensor of proposed points in original space [n_evals, dim]
        Y_turbo: torch.Tensor of objective values [n_evals, 1] (or None for pending)
        trial_status: dict mapping trial_index to "completed" or "pending"
        worst_dmerc_by_trial: dict mapping completed trial_index to worst_dmerc
    """
    if not proc0:
        return
    
    # Convert tensors to numpy for CSV export
    X_np = X_turbo.detach().cpu().numpy()
    Y_np = Y_turbo.detach().cpu().numpy() if Y_turbo is not None else None
    
    # Build DataFrame with column order: trial_index, objective, status, then parameter columns
    rows = []
    for i in range(X_np.shape[0]):
        row = {"trial_index": i}
        
        # Add objective value if completed
        if Y_np is not None and i < len(Y_np):
            if trial_status.get(i) == "completed":
                row["loss"] = Y_np[i, 0]
            else:
                row["loss"] = np.nan
        else:
            row["loss"] = np.nan

        if trial_status.get(i) == "completed" and worst_dmerc_by_trial is not None:
            row["worst_dmerc"] = worst_dmerc_by_trial.get(i, np.nan)
        else:
            row["worst_dmerc"] = np.nan
        
        # Add status
        row["status"] = trial_status.get(i, "pending")
        
        # Add parameter values
        for j, name in enumerate(param_names):
            row[name] = X_np[i, j]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_file + ".tmp", index=False)
    os.replace(csv_file + ".tmp", csv_file)



def manager_process(max_evals=100):
    """Manager process (rank 0) that coordinates TuRBO optimization."""
    # Initialize TuRBO state
    turbo_state = TurboState(dim=dim, batch_size=batch_size)
    log(f"Initialized TuRBO state: {turbo_state}")

    # Data storage
    X_turbo_list = []  # List of numpy arrays in original space (only appended when sent)
    Y_turbo_list = []  # List of objective values (in order of completion)
    trial_status = {}  # trial_index -> "completed" or "pending"
    active_trials = {}  # trial_index -> worker_rank
    trial_batch = []  # Queue of (trial_index, params, x_original) to send
    all_X = {}  # mapping trial_index -> x (original space) for all enqueued trials
    results = {}  # mapping trial_index -> objective value for completed trials
    worst_dmerc_results = {}  # mapping trial_index -> worst_dmerc for completed trials
    next_trial_index = 0
    completed_trials = 0
    accumulated_generation_time = 0.0

    log(f"Starting TuRBO optimization with {size-1} worker processes, max_evals={max_evals}")

    # Generate initial batch using Sobol sequence and enqueue into trial_batch
    generation_start_time = time.time()
    sobol = SobolEngine(dim, scramble=True, seed=0)
    # Make sure there are at least as many points as n_workers * 2:
    n_init = max(2 * dim, (size - 1) * 2)
    X_init_normalized = sobol.draw(n=min(n_init, max_evals)).to(dtype=dtype, device=device)

    # Convert to original space and enqueue
    X_init = unnormalize_point(X_init_normalized.cpu().numpy(), lb, ub)
    for i in range(X_init.shape[0]):
        x = X_init[i]
        params = {param_names[j]: float(x[j]) for j in range(dim)}
        trial_batch.append((next_trial_index, params, x))
        all_X[next_trial_index] = x
        next_trial_index += 1

    generation_time = time.time() - generation_start_time
    accumulated_generation_time += generation_time
    log(f"Enqueued {len(X_init)} initial points using Sobol sequence ({generation_time:.2f} seconds)")

    # Send initial work to workers from trial_batch
    for worker_rank in range(1, size):
        if not trial_batch or completed_trials >= max_evals:
            break
        trial_index, params, x = trial_batch.pop(0)
        active_trials[trial_index] = worker_rank
        trial_status[trial_index] = "pending"
        X_turbo_list.append(x)
        comm.send((params, trial_index), dest=worker_rank, tag=WORK_TAG)
        log(f"Sent initial trial {trial_index} to worker {worker_rank}")

    log(f"Remaining enqueued points to be sent: {len(trial_batch)}")

    # Main optimization loop
    while active_trials or completed_trials < max_evals:
        if completed_trials >= max_evals:
            log("Reached max_evals limit, stopping optimization.")
            break

        # Check for wallclock time limit
        elapsed_minutes = (time.time() - start_time) / 60.0
        if elapsed_minutes >= max_wallclock_minutes:
            log(f"Reached max wallclock time limit of {max_wallclock_minutes} minutes, stopping optimization.")
            break

        # If our send-queue is empty, try to generate a new batch (concurrently while workers are running)
        if not trial_batch and completed_trials < max_evals:
            # Only generate GP-based batches if we have enough completed data
            n_completed = sum(1 for s in trial_status.values() if s == "completed")
            min_completed_for_gp = max(batch_size + 1, 4)  # At least 4 points or batch_size + 1

            if n_completed >= min_completed_for_gp:
                generation_start_time = time.time()

                # Prepare data for GP from completed trials using stored mappings
                completed_indices = sorted([i for i, s in trial_status.items() if s == "completed"])
                if len(completed_indices) > 0:
                    # Gather X and Y for completed trials in a consistent order
                    X_completed_np = np.array([all_X[i] for i in completed_indices])
                    Y_completed_values = np.array([results[i] for i in completed_indices])

                    # Normalize to [0, 1] for GP
                    X_normalized = normalize_point(X_completed_np, lb, ub)
                    X_torch = torch.from_numpy(X_normalized).to(dtype=dtype, device=device)
                    Y_torch = torch.from_numpy(Y_completed_values.reshape(-1, 1)).to(dtype=dtype, device=device)

                    # Fit GP and generate new batch
                    with gpytorch.settings.max_cholesky_size(float("inf")):
                        model = fit_gp_model(X_torch, Y_torch)
                        X_next_normalized = generate_turbo_batch(
                            state=turbo_state,
                            model=model,
                            X_turbo=X_torch,
                            Y_turbo=(Y_torch - Y_torch.mean()) / Y_torch.std(),
                            batch_size=batch_size,
                            n_candidates=min(5000, max(2000, 200 * dim)),
                            num_restarts=10,
                            raw_samples=512,
                            # acqf="ts",
                            acqf="ei",
                        )

                    X_next = unnormalize_point(X_next_normalized.detach().cpu().numpy(), lb, ub)

                    generation_time = time.time() - generation_start_time
                    accumulated_generation_time += generation_time
                    log(f"Time to generate batch: {generation_time:.2f} seconds (accumulated: {accumulated_generation_time:.2f} seconds)")

                    # Enqueue generated points and record them in all_X
                    for x in X_next:
                        params = {param_names[j]: float(x[j]) for j in range(dim)}
                        trial_batch.append((next_trial_index, params, x))
                        all_X[next_trial_index] = x
                        next_trial_index += 1
                    log(f"Generated {len(X_next)} new points via TuRBO, now {len(trial_batch)} points enqueued")

                    # Update TuRBO state
                    try:
                        turbo_state = update_turbo_state(turbo_state, np.array(Y_completed_values))
                        log(f"Updated TuRBO state: best_value={turbo_state.best_value:.2e}, length={turbo_state.length:.2e}")
                    except Exception:
                        pass

        # Non-blocking check for results
        status = MPI.Status()
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status):
            worker_rank = status.Get_source()
            trial_index, result_value = comm.recv(source=worker_rank, tag=RESULT_TAG)

            log(f"Completed trial {trial_index} from worker {worker_rank}: {result_value}")

            # Mark trial as completed
            trial_status[trial_index] = "completed"
            results[trial_index] = result_value["loss"]
            worst_dmerc_results[trial_index] = result_value["worst_dmerc"]
            Y_turbo_list.append(result_value["loss"])
            completed_trials += 1

            # Remove from active trials
            if trial_index in active_trials:
                del active_trials[trial_index]

            # Save state periodically (ordered by trial index)
            if completed_trials % save_frequency == 0:
                # Build full arrays of shape [next_trial_index, dim] and [next_trial_index, 1]
                n_total = max(1, next_trial_index)
                X_full = np.full((n_total, dim), np.nan, dtype=np.float64)
                Y_full = np.full((n_total, 1), np.nan, dtype=np.float64)
                for idx, x in all_X.items():
                    if 0 <= idx < n_total:
                        X_full[idx] = x
                for idx, val in results.items():
                    if 0 <= idx < n_total:
                        Y_full[idx, 0] = val

                X_tensor = torch.from_numpy(X_full).to(dtype=dtype)
                Y_tensor = torch.from_numpy(Y_full).to(dtype=dtype)
                save_turbo_state(X_tensor, Y_tensor, trial_status, worst_dmerc_results)

            # Send next trial from queue to this worker if available
            if trial_batch and completed_trials < max_evals:
                next_trial_index_to_send, params, x = trial_batch.pop(0)
                active_trials[next_trial_index_to_send] = worker_rank
                trial_status[next_trial_index_to_send] = "pending"
                X_turbo_list.append(x)
                comm.send((params, next_trial_index_to_send), dest=worker_rank, tag=WORK_TAG)
                log(f"Sent trial {next_trial_index_to_send} to worker {worker_rank} ({len(trial_batch)} enqueued remaining)")
            else:
                # If no queued work, inform worker to wait (we simply don't send; worker will probe again)
                log(f"No queued work to send to worker {worker_rank}")

        else:
            # No result available; small sleep to prevent busy waiting
            time.sleep(0.01)

    # Send stop signals to all workers
    for worker_rank in range(1, size):
        try:
            comm.send(None, dest=worker_rank, tag=STOP_TAG)
        except:
            pass

    # Save final state ordered by trial index
    n_total = max(1, next_trial_index)
    X_full = np.full((n_total, dim), np.nan, dtype=np.float64)
    Y_full = np.full((n_total, 1), np.nan, dtype=np.float64)
    for idx, x in all_X.items():
        if 0 <= idx < n_total:
            X_full[idx] = x
    for idx, val in results.items():
        if 0 <= idx < n_total:
            Y_full[idx, 0] = val

    X_tensor = torch.from_numpy(X_full).to(dtype=dtype)
    Y_tensor = torch.from_numpy(Y_full).to(dtype=dtype)
    save_turbo_state(X_tensor, Y_tensor, trial_status, worst_dmerc_results)

    log(f"Optimization complete. Completed {completed_trials} trials.")

    return X_turbo_list, Y_turbo_list, trial_status


def worker_process():
    """Worker process that evaluates the objective function"""
    log(f"Worker {rank} starting")
    accumulated_waiting_time = 0.0
    while True:
        # Check for work or stop signal
        status = MPI.Status()
        probe_start_time = time.time()
        comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)
        probe_time = time.time() - probe_start_time
        accumulated_waiting_time += probe_time
        log(f"Worker {rank} probe time: {probe_time:.4f} seconds (accumulated: {accumulated_waiting_time:.2f} seconds)")
        tag = status.Get_tag()
        
        if tag == STOP_TAG:
            comm.recv(source=0, tag=STOP_TAG)
            log(f"Worker {rank} stopping")
            break
        elif tag == WORK_TAG:
            parameters, trial_index = comm.recv(source=0, tag=WORK_TAG)
            log(f"Worker {rank} received trial {trial_index}")
            
            # Evaluate objective function
            result_value = run_objective(parameters, trial_index)
            
            # Send result back to manager
            comm.send((trial_index, result_value), dest=0, tag=RESULT_TAG)
            log(f"Worker {rank} completed trial {trial_index}: {result_value}")

def main():
    if not proc0:
        # Worker process
        worker_process()
        return
    
    # Manager process
    log("Starting TuRBO optimization manager")
    
    # Run manager process
    X_turbo_list, Y_turbo_list, trial_status = manager_process(max_evals=num_evals)
    
    if len(Y_turbo_list) == 0:
        log("No evaluations completed!")
        return
    
    # Convert to numpy for analysis
    X_all = np.array(X_turbo_list)
    Y_all = np.array(Y_turbo_list)
    
    # Find best evaluation
    best_idx = np.argmin(Y_all)
    best_value = Y_all[best_idx]
    best_x = X_all[best_idx]
    
    log(f"Best value found: {best_value}")
    log(f"Best parameters: {dict(zip(param_names, best_x))}")
    
    # Evaluate best point
    best_params = {param_names[i]: best_x[i] for i in range(dim)}
    run_objective(best_params, 999999)
    save_vmec_files("optimized")
    
    # Final CSV already saved by manager (ordered by trial index)
    log(f"Final CSV (ordered by trial index) saved to {csv_file} by manager")

if __name__ == "__main__":
    main()
