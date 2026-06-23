#!/usr/bin/env python

import os
import pickle
import json
import time
import traceback
import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from vmecpp.simsopt_compat import Vmec
import warnings

from alpha_opt import SurfaceGarabedianQuantiles, get_worst_DMerc_normalized
from alpha_opt.objective import get_objective
from alpha_opt.gpu_tracing import compute_alpha_loss
from mpi4py import MPI

# from generation_strategy import my_generation_strategy

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

n_particles = 25000
t_max = 1e-1
tau = 0.1

# Trace only until losses exceed this fraction:
maxloss = 0.02
t_block = 1e-3

# See 20251109-02 GPU tracing - convergence with respect to tol and min_dt.docx
tol = 1e-6
min_dt = 1e-9

# # Maximum mode numbers to vary in the SurfaceRZFourier representation.
# m_max = 1
# n_max = 1

data_file = "/pscratch/sd/l/landrema/20260402-01-alpha_optimization_unconstrained_parameter_scans/20260402-01_prepare_weighted_data_nfpAtLeast3_Garabedian.h5"

mpol = 2
ntor = 2
# Range of each dof will be [x_min, 1 - x_min]
x_min = 0.1

# How to handle cases in which VMEC does not converge:
# If False, Ax will classify the run as FAILED.
# If True, Ax will classify the run as COMPLETED with a large value of the objective.
treat_failures_as_big_number = True
fail_val = 5.5  # Like losing maxloss of the energy at 10^{-5.5} sec.
# If the Mercier constraint evaluation fails, return this value (should be less than 0, so it violates the constraint)
DMerc_fail_val = -0.5
# DMerc_fail_val = -np.inf

# Max number of evaluations to perform
num_evals = 10000

# Number of trials to generate at once in batches
batch_size = 1

max_wallclock_minutes = 60 * 2 - 5  # Leave 5 minutes buffer at end of job

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

# surface, dim_x, x_scale, x0 = init_optimizable_surface(
#     m_max,
#     n_max,
#     vmec.indata.nfp,
#     major_radius,
#     minor_radius,
#     scale=False,
#     verbose=proc0,
# )
surface = SurfaceGarabedianQuantiles(
    vmec.indata.nfp,
    mpol=mpol,
    ntor=ntor,
    minor_radius=minor_radius,
    major_radius=major_radius,
    filename=data_file,
    exact_radii=True,
)
dim_x = len(surface.x)
x_scale = np.ones(dim_x)

#qs = QuasisymmetryRatioResidual(
#    vmec, 
#    np.linspace(0, 1, 11),
#    helicity_m=1,
#    helicity_n=1,
#)
#
#wrapped_objective = get_objective(vmec, surface, x_scaxle, qs.total, fail_val=fail_val)

def raw_objective():
    wout_filename = "wout_tmp.nc"
    vmec.wout.save(wout_filename)
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

# bounds_data = np.loadtxt(
#     # "/Users/mattland/Box/work25/20250912-02-get_bounds_on_Garabedian_Deltas.dat",
#     "../20250912-02-get_bounds_on_Garabedian_Deltas.dat",
#     skiprows=1,
#     delimiter=",",
# )
# if proc0:
#     log(bounds_data)
# # columns are m, n, min, max

# print("Initial objective:", wrapped_objective(x0))

# Record start time for elapsed time tracking
start_time = time.time()

# Define parameter space
# param_names = [f"x{i}" for i in range(dim_x)]
param_names = [x.replace(",", "_") for x in surface.local_dof_names]
param_bounds = [
    RangeParameterConfig(
        name=name,
        parameter_type="float",
        #bounds=(-parameter_max, parameter_max)
        bounds=(x_min, 1.0 - x_min),
    )
    for name in param_names
]

# margin = 0
# n_rows = bounds_data.shape[0]
# for name in param_names:
#     hit = False
#     for j in range(n_rows):
#         m, n, data_min, data_max = bounds_data[j, :]
#         if name == f"Delta({round(m)},{round(n)})":
#             if proc0:
#                 log("hit found for", name)
#             hit = True
#             # Scale min and max for this aspect ratio
#             data_min = float(data_min) * minor_radius
#             data_max = float(data_max) * minor_radius
#             absolute_margin = margin * (data_max - data_min)
#             # param_bounds.append(
#             #     {"name": name, "type": "range", "bounds": [data_min - absolute_margin, data_max + absolute_margin]}
#             # )
#             param_bounds.append(
#                 RangeParameterConfig(
#                     name=name,
#                     parameter_type="float",
#                     bounds=(data_min - absolute_margin, data_max + absolute_margin)
#                 )
#             )
#     if not hit:
#         raise RuntimeError(f"dof {name} was not found in the saved list of bounds")

# # optimum from simsopt least-squares solver:
# x_simsopt = [2.8728233965802873e-03,  1.7388588039722933e-03,  8.9469110105193698e-04,  8.8404973186767211e-02, -3.1783427792145177e-05, -7.6714341515670536e-03, -8.4536421286537669e-02]
# margin = 2
# for j, name in enumerate(param_names):
#     # param_bounds.append({"name": name, "type": "range", "bounds": [-abs(x_simsopt[j]) * margin, abs(x_simsopt[j]) * margin]})
#     param_bounds.append(
#         RangeParameterConfig(
#             name=name,
#             parameter_type="float",
#             bounds=(-abs(x_simsopt[j]) * margin, abs(x_simsopt[j]) * margin)
#         )
#     )

if proc0:
    log(param_bounds)

# MPI tags for different message types
WORK_TAG = 1
RESULT_TAG = 2
STOP_TAG = 3

# Suppress the specific BoTorch/Ax numerical warning about very small noise values.
# We only ignore the exact message to avoid hiding other NumericalWarnings.
warnings.filterwarnings(
    "ignore",
    message=r"Very small noise values detected\. This will likely lead to numerical instabilities\. Rounding small noise values up to 1e-06\."
)

def run_objective(params, trial_index):
    """Evaluate the objective function and constraint for given parameters"""
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
        
        # Evaluate the Mercier constraint
        worst_dmerc = get_worst_DMerc_normalized(vmec, 0.2, 0.95)
        
        objective_time = time.time() - objective_start_time
        log(f"Objective evaluation for eval {trial_index} took {objective_time:.1f} seconds")
        log(f"  Loss: {val}, Worst DMerc: {worst_dmerc}")

        with open("results.txt", "a") as f:
            f.write(f"worst_DMerc = {worst_dmerc}\n")
        
        # Return both the objective value and the constraint
        return {"loss": val, "worst_dmerc": worst_dmerc}
    except Exception as e:
        objective_time = time.time() - objective_start_time
        log(f"Objective evaluation failed after {objective_time:.1f} seconds. {e}")
        log(f"Full traceback:\n{traceback.format_exc()}")
        return {"loss": fail_val, "worst_dmerc": DMerc_fail_val}  # Return constraint failure value
    finally:
        # Always return to original directory
        os.chdir(original_cwd)

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

def save_ax_state(ax_client):
    """Save AxClient state to JSON & CSV files (atomic rename)."""
    ax_client.save_to_json_file(state_file + ".tmp")
    # df = exp_to_df(ax_client.experiment)
    df = ax_client.summarize()

    worst_dmerc_values = []
    for trial_index in df['trial_index']:
        try:
            trial_data = ax_client._experiment.lookup_data(trial_indices=[int(trial_index)])
            trial_df = trial_data.df
            worst_dmerc_row = trial_df[trial_df['metric_name'] == 'worst_dmerc']
            if not worst_dmerc_row.empty:
                worst_dmerc_values.append(worst_dmerc_row['mean'].values[0])
            else:
                worst_dmerc_values.append(None)
        except Exception as e:
            log(f"Warning: Could not extract worst_dmerc for trial {trial_index}: {e}")
            worst_dmerc_values.append(None)

    df['worst_dmerc'] = worst_dmerc_values
    df.to_csv(csv_file + ".tmp")
    os.replace(state_file + ".tmp", state_file)
    os.replace(csv_file + ".tmp", csv_file)

def manager_process(ax_client, max_trials=100):
    """Manager process (rank 0) that coordinates the optimization"""
    active_trials = {}  # trial_index -> rank mapping
    completed_trials = 0
    accumulated_generation_time = 0.0
    
    # Batch trial management
    trial_batch = []  # List of (trial_index, parameters) tuples
    
    log(f"Starting optimization with {size-1} worker processes, max_trials={max_trials}")
    
    # Generate initial batch and send work to all workers
    generation_start_time = time.time()
    n_initial_trials = (size - 1) * 2  # Number of workers times 2
    # n_initial_trials = max(50, (size - 1) * 2)  # Number of workers times 2
    trials = ax_client.get_next_trials(n_initial_trials)
    trial_batch = list(trials.items())  # Convert to list of (trial_index, parameters) tuples
    generation_time = time.time() - generation_start_time
    accumulated_generation_time += generation_time
    log(f"Time to generate initial batch of {len(trial_batch)} trials: {generation_time:.2f} seconds")
    
    # Send work to initial workers
    for worker_rank in range(1, size):
        trial_index, parameters = trial_batch.pop(0)
        active_trials[trial_index] = worker_rank
        comm.send((parameters, trial_index), dest=worker_rank, tag=WORK_TAG)
        log(f"Sent trial {trial_index} to worker {worker_rank}")
    
    # Main optimization loop - process results as they come in
    # while active_trials and completed_trials < max_trials:
    while active_trials:
        if completed_trials >= max_trials:
            log("Reached max_trials limit, stopping optimization.")
            break
        # Check for wallclock time limit
        elapsed_minutes = (time.time() - start_time) / 60.0
        if elapsed_minutes >= max_wallclock_minutes:
            log(f"Reached max wallclock time limit of {max_wallclock_minutes} minutes, stopping optimization.")
            break

        # If batch is empty, generate a new batch
        if not trial_batch:
            try:
                log("No trials left in batch, generating new batch")
                generation_start_time = time.time()
                trials = ax_client.get_next_trials(batch_size)
                trial_batch = list(trials.items())
                generation_time = time.time() - generation_start_time
                accumulated_generation_time += generation_time
                log(f"Time to generate new batch of {len(trial_batch)} trials: {generation_time:.2f} seconds (accumulated: {accumulated_generation_time:.2f} seconds)")
            except Exception as e:
                # No more trials available
                log(f"No more trials available: {e}")

        # Non-blocking check for results from any worker
        status = MPI.Status()
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status):
            worker_rank = status.Get_source()
            trial_index, result_value = comm.recv(source=worker_rank, tag=RESULT_TAG)
            
            # Complete the trial in Ax
            try:
                if result_value["loss"] == fail_val and (not treat_failures_as_big_number):
                    ax_client.log_trial_failure(trial_index=trial_index)
                else:
                    # ax_client.complete_trial(trial_index=trial_index,
                    # raw_data=result_value)
                    # If the SEM is reported as 0, there are many warnings
                    # printed by Ax: "Very small noise values detected. This
                    # will likely lead to numerical instabilities. Rounding
                    # small noise values up to 1e-06."
                    if specify_zero_variance:
                        ax_client.complete_trial(trial_index=trial_index, raw_data={"loss": (result_value["loss"], 0.0), "worst_dmerc": (result_value["worst_dmerc"], 0.0)})  # last number is standard error of the mean
                    else:
                        ax_client.complete_trial(trial_index=trial_index, raw_data={"loss": result_value["loss"], "worst_dmerc": result_value["worst_dmerc"]})

                log(f"Completed trial {trial_index} from worker {worker_rank}: loss={result_value['loss']}, worst_dmerc={result_value['worst_dmerc']}")
                completed_trials += 1
                if (completed_trials + 1) % save_frequency == 0:
                    save_ax_state(ax_client)
                
                # Remove from active trials
                del active_trials[trial_index]
                
                # Save history after each completed trial
                # with open(history_file, "wb") as f:
                #     pickle.dump(ax_client, f)
                # ax_client.save_to_json_file(state_file)
                                
                # Send next trial from batch to worker
                if trial_batch:
                    trial_index, parameters = trial_batch.pop(0)
                    active_trials[trial_index] = worker_rank
                    comm.send((parameters, trial_index), dest=worker_rank, tag=WORK_TAG)
                    log(f"Sent trial {trial_index} to worker {worker_rank} (batch has {len(trial_batch)} remaining)")
                else:
                    # Batch is empty and no more trials available
                    log(f"No more work for worker {worker_rank}")
                    comm.send(None, dest=worker_rank, tag=STOP_TAG)
                    
            except Exception as e:
                log(f"Error completing trial {trial_index}: {e}")
                if trial_index in active_trials:
                    del active_trials[trial_index]
        
        # Small sleep to prevent busy waiting
        time.sleep(0.01)
    
    # Send stop signals to any remaining workers
    for worker_rank in range(1, size):
        try:
            comm.send(None, dest=worker_rank, tag=STOP_TAG)
        except:
            pass  # Worker may have already stopped
    
    log(f"Optimization complete. Completed {completed_trials} trials.")

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
    
    # Manager process - load or create AxClient
    # ax_client = AxClient()
    if os.path.exists(state_file):
        # ax_client.load_from_json_file(state_file)
        ax_client = Client.load_from_json_file(state_file)
        # with open(history_file, "rb") as f:
        #     ax_client = pickle.load(f)
        log("Loaded previous optimization history.")
        # print(f"Previous trials completed: {len(ax_client.experiment.trials)}")
    else:
        log("Creating new Ax experiment")
        ax_client = Client()
        # ax_client.create_experiment(
        #     name="my_experiment",
        #     parameters=param_bounds,
        # )
        ax_client.configure_experiment(param_bounds)
        # Configure optimization with Mercier stability constraint: worst_dmerc >= 0 (positive)
        ax_client.configure_optimization(
            objective="-loss",
            # outcome_constraints=["worst_dmerc >= 0"],
        )
        # ax_client.set_generation_strategy(my_generation_strategy("qUCB", beta=0.01))
        #ax_client.set_generation_strategy(my_generation_strategy("qPI"))

        # # Add x0 as an initial evaluation point
        # x0_params = {name: x0[i] for i, name in enumerate(param_names)}
        # parameters, trial_index = ax_client.attach_trial(x0_params)
        # ax_client.complete_trial(trial_index=trial_index, raw_data=wrapped_objective(x0))
        # print(f"Added x0 as initial evaluation point (trial {trial_index})")
    
    # Run manager process
    manager_process(ax_client, max_trials=num_evals)
    
    # Save final results
    save_ax_state(ax_client)

    df = ax_client.summarize()
    log(df)

    log(f"Best parameters found:")
    parameters_of_best_model, prediction, index_for_best_model, name = ax_client.get_best_parameterization()
    log(f"  Prediction (mean, variance): {prediction}")
    log(f"  Index: {index_for_best_model}")
    log(f"  Parameters: {parameters_of_best_model}")
    log(f"  Name: {name}")
    run_objective(parameters_of_best_model, 999998)
    save_vmec_files("best_model")

    best_index = df["loss"].idxmin()
    best_row = df.loc[best_index]
    log(f"  Lowest objective: {best_row['loss']}")
    log(f"  Index: {best_index}")
    best_parameters = {name: best_row[name] for name in param_names}
    log("  Parameters:", best_parameters)
    run_objective(best_parameters, 999999)
    save_vmec_files("optimized")
    np.testing.assert_allclose(df["loss"].min(), best_row["loss"])

if __name__ == "__main__":
    main()
