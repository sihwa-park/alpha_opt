#!/usr/bin/env python
"""Synchronous BoTorch optimization with qLogNEI (q=1) for stellarator
alpha-loss minimization.

Usage:
    python botorch_optimize.py [config.yaml]
"""

import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional
import random

import numpy as np
import torch
import wandb
import yaml
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.analytic import LogExpectedImprovement, LogNoisyExpectedImprovement, LogConstrainedExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from vmecpp.simsopt_compat import Vmec
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.utils.sampling import optimize_posterior_samples
from botorch.sampling.pathwise import draw_matheron_paths

from alpha_opt.mercier.mercier import get_worst_DMerc_normalized
from alpha_opt.objective import VmecConvergenceError, get_objective
from alpha_opt.surface import SurfaceGarabedianQuantiles, SurfaceGarabedianLinear
from alpha_opt.tracing import compute_alpha_loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ["t_max", "tau", "tol", "max_B_target", "x_min"]:
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])
    return cfg


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Outcome of a single objective evaluation."""
    loss: Optional[float]
    worst_dmerc: Optional[float]
    vmec_failed: bool
    tracing_failed: bool
    case: Optional[int]
    var_obj: Optional[float]

    @property
    def failed(self) -> bool:
        return self.vmec_failed or self.tracing_failed


# ---------------------------------------------------------------------------
# Objective builder
# ---------------------------------------------------------------------------

def build_objective(cfg, vmec, surface, phiedge_high):
    x_scale = np.ones(len(surface.x))

    def raw_objective():
        wout_filename = "wout_tmp.nc"
        vmec.wout.save(wout_filename)
        return compute_alpha_loss(
            wout_filename,
            n_particles=cfg["n_particles"],
            t_max=cfg["t_max"],
            tau=cfg["tau"],
            t_block=cfg["t_block"],
            min_dt=cfg["min_dt"],
            maxloss=cfg["maxloss"],
            tol=cfg["tol"],
            vacuum=cfg["vacuum"],
        )

    return get_objective(
        vmec,
        surface,
        x_scale,
        raw_objective,
        max_B=cfg["max_B_target"],
        max_B_iterations=cfg["max_B_iterations"],
        phiedge=phiedge_high,
    )


# ---------------------------------------------------------------------------
# Single trial evaluation
# ---------------------------------------------------------------------------

def evaluate(x, objective, vmec, cfg, trial_index, run_name) -> EvalResult:
    """Run one trial. Returns an EvalResult — never raises."""
    eval_dir = f"evals/{run_name}_eval{trial_index:06d}"
    os.makedirs(eval_dir, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(eval_dir)
    print(f"Evaluating trial {trial_index} with x: "
          f"{np.array2string(np.asarray(x), precision=6)}")
    t0 = time.time()
    try:
        loss, case, var_obj = objective(np.asarray(x))
        worst_dmerc = get_worst_DMerc_normalized(vmec, 0.2, 0.95)
        dt = time.time() - t0
        print(f"Trial {trial_index} done in {dt:.1f}s: "
              f"loss={loss:.6e}, worst_dmerc={worst_dmerc:.6e}")
        with open("results.txt", "a") as f:
            f.write(f"worst_DMerc = {worst_dmerc}\n")
        return EvalResult(
            loss=float(loss),
            worst_dmerc=float(worst_dmerc),
            vmec_failed=False,
            tracing_failed=False,
            case=case,
            var_obj=var_obj,
        )

    except VmecConvergenceError as e:
        dt = time.time() - t0
        print(f"Trial {trial_index} VMEC FAILED after {dt:.1f}s: {e}")
        return EvalResult(
            loss=None,
            worst_dmerc=None,
            vmec_failed=True,
            tracing_failed=False,
            case=None,
            var_obj=None,
        )

    except Exception as e:
        dt = time.time() - t0
        print(f"Trial {trial_index} TRACING FAILED after {dt:.1f}s: {e}")
        print(traceback.format_exc())
        return EvalResult(
            loss=None,
            worst_dmerc=None,
            vmec_failed=False,
            tracing_failed=True,
            case=None,
            var_obj=None,
        )
    finally:
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# GP proposal
# ---------------------------------------------------------------------------

def propose_next(X_t, Y_t, Y_var, bounds, cfg, Xc_t=None, C_t=None) -> tuple[np.ndarray, float, dict, float]:
    """Fit a GP on successful trials and return the next candidate."""
    device = bounds.device
    if cfg["kernel"] == "matern":
        use_rbf = False
    elif cfg["kernel"] == "rbf":
        use_rbf = True
    if not cfg["use_gp_constraints"]:
        model = SingleTaskGP(
            X_t,
            (-Y_t).unsqueeze(-1),  # BoTorch's acquisition functions assume a maximization problem, so we negate the loss.
            train_Yvar=Y_var.unsqueeze(-1),  # small noise to improve numerical stability
            covar_module=get_covar_module_with_dim_scaled_prior(ard_num_dims=X_t.shape[-1], use_rbf_kernel=use_rbf),
            # covar_module=ScaleKernel(RBFKernel(ard_num_dims=X_t.shape[-1])),
            input_transform=Normalize(d=X_t.shape[-1], bounds=bounds),
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        gp_hparams = {}
        for j, ls in enumerate(
            model.covar_module.lengthscale.detach().flatten().tolist()
        ):
            gp_hparams[f"gp/lengthscale_{j}"] = ls
    else:
        gp_obj = SingleTaskGP(
            X_t,
            (-Y_t).unsqueeze(-1),  # BoTorch's acquisition functions assume a maximization problem, so we negate the loss.
            train_Yvar=Y_var.unsqueeze(-1),  # small noise to improve numerical stability
            covar_module=get_covar_module_with_dim_scaled_prior(ard_num_dims=X_t.shape[-1], use_rbf_kernel=use_rbf),
            # covar_module=ScaleKernel(RBFKernel(ard_num_dims=X_t.shape[-1])),
            input_transform=Normalize(d=X_t.shape[-1], bounds=bounds),
            outcome_transform=Standardize(m=1),
        )
        gp_con = SingleTaskGP(
            Xc_t,
            C_t.unsqueeze(-1),  # constraint values
            covar_module=get_covar_module_with_dim_scaled_prior(ard_num_dims=X_t.shape[-1], use_rbf_kernel=use_rbf),
            input_transform=Normalize(d=X_t.shape[-1], bounds=bounds),
            outcome_transform=None,
        )
        model = ModelListGP(gp_obj, gp_con)
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        gp_hparams = {}
        for j, ls in enumerate(
            model.models[0].covar_module.lengthscale.detach().flatten().tolist()
        ):
            gp_hparams[f"gp/lengthscale_{j}"] = ls
    
    if cfg["acquisition"] == "ei":
        acq = LogNoisyExpectedImprovement(
            model=model,
            X_observed=X_t,
        )
        candidate, acq_val = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=cfg["num_restarts"],
            raw_samples=cfg["raw_samples"],
        )
        with torch.no_grad():
            candidate_variance = model.posterior(candidate).variance.squeeze().item()
        return candidate.squeeze(0), float(acq_val.item()), gp_hparams, candidate_variance
    elif cfg["acquisition"] == "cei":
        acq = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_t,
            prune_baseline=cfg["prune_baseline"],
            objective=GenericMCObjective(lambda Z, X=None: Z[..., 0]),
            constraints=[lambda Z: Z[..., 1]],  # specify that the second model's output (the constraint) should be <= 0
        )
        candidate, acq_val = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=cfg["num_restarts"],
            raw_samples=cfg["raw_samples"],
        )
        with torch.no_grad():
            candidate_variance = model.models[0].posterior(candidate).variance.squeeze().item()
        return candidate.squeeze(0), float(acq_val.item()), gp_hparams, candidate_variance
    elif cfg["acquisition"] == "ts":
        paths = draw_matheron_paths(model, sample_shape=torch.Size([1]))
        optimal_input, optimal_output = optimize_posterior_samples(paths=paths, bounds=bounds, num_restarts=cfg["num_restarts"], raw_samples=cfg["raw_samples"])
        with torch.no_grad():
            candidate_variance = model.posterior(optimal_input).variance.squeeze().item()
        return optimal_input.squeeze(0), float(optimal_output.item()), gp_hparams, candidate_variance
    else:
        raise ValueError(f"Unsupported acquisition function: {cfg['acquisition']}")

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_state(state_file, csv_file, param_names, X_np, Y_np, Y_var_np, results):
    state = {
        "param_names": list(param_names),
        "X": X_np.tolist(),
        "Y": Y_np.tolist(),
        "Y_var": Y_var_np.tolist(),
        "results": [
            {
                "loss": r.loss,
                "worst_dmerc": r.worst_dmerc,
                "vmec_failed": r.vmec_failed,
                "tracing_failed": r.tracing_failed,
            }
            for r in results
        ],
    }
    with open(state_file + ".tmp", "w") as f:
        json.dump(state, f, indent=2)
    os.replace(state_file + ".tmp", state_file)

    with open(csv_file + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["trial_index"] + list(param_names)
            + ["loss", "loss_var", "worst_dmerc", "vmec_failed", "tracing_failed"]
        )
        for i, (x, y, y_var, r) in enumerate(zip(X_np, Y_np, Y_var_np, results)):
            w.writerow(
                [i] + list(x)
                + [y, y_var, r.worst_dmerc, r.vmec_failed, r.tracing_failed]
            )
    os.replace(csv_file + ".tmp", csv_file)

def save_state_with_c(state_file, csv_file, param_names, X_np, Y_np, Y_var_np, Xc_np, C_np, results):
    state = {
        "param_names": list(param_names),
        "X": X_np.tolist(),
        "Y": Y_np.tolist(),
        "Xc": Xc_np.tolist(),
        "C": C_np.tolist(),
        "Y_var": Y_var_np.tolist(),
        "results": [
            {
                "loss": r.loss,
                "worst_dmerc": r.worst_dmerc,
                "vmec_failed": r.vmec_failed,
                "tracing_failed": r.tracing_failed,
            }
            for r in results
        ],
    }
    with open(state_file + ".tmp", "w") as f:
        json.dump(state, f, indent=2)
    os.replace(state_file + ".tmp", state_file)

    with open(csv_file + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["trial_index"] + list(param_names)
            + ["loss", "loss_var", "worst_dmerc", "vmec_failed", "tracing_failed"]
        )
        for i, (x, y, y_var, r) in enumerate(zip(X_np, Y_np, Y_var_np, results)):
            w.writerow(
                [i] + list(x)
                + [y, y_var, r.worst_dmerc, r.vmec_failed, r.tracing_failed]
            )
    os.replace(csv_file + ".tmp", csv_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cpu")
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(cfg_path)
    print(f"Loaded config from {cfg_path}")
    torch.set_default_dtype(torch.double)
    
    wandb_run = wandb.init(
        entity="sp2582-cornell-university",
        project=cfg["wandb_project"],
        config=cfg,
    )
    run_name = wandb_run.name
    cfg["seed"] = wandb_run.config.get("seed", 0)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    # --- Geometry ---
    aspect_ratio = cfg["aspect_ratio"]
    minor_radius = (
        float(cfg["minor_radius"])
        if cfg.get("minor_radius") is not None
        else 3.1 / aspect_ratio ** 0.38
    )
    major_radius = minor_radius * aspect_ratio
    print(f"aspect_ratio={aspect_ratio}, minor_radius={minor_radius:.6f}, "
          f"major_radius={major_radius:.6f}")

    # --- VMEC ---
    vmec = Vmec(cfg["vmec_input_file"], verbose=True)
    avg_B_estimate = cfg["max_B_target"] / np.sqrt(2)
    phiedge_high = np.pi * avg_B_estimate * minor_radius ** 2 * 2
    vmec.set("phiedge", phiedge_high)

    # --- Surface ---
    if cfg["parameterization"] == "garabedian_quantiles":
        surface = SurfaceGarabedianQuantiles(
            vmec.indata.nfp,
            mpol=cfg["mpol"],
            ntor=cfg["ntor"],
            minor_radius=minor_radius,
            major_radius=major_radius,
            filename=cfg["data_file"],
            exact_radii=True,
        )
    elif cfg["parameterization"] == "garabedian_linear":
        surface = SurfaceGarabedianLinear(
            vmec.indata.nfp,
            mpol=cfg["mpol"],
            ntor=cfg["ntor"],
            minor_radius=minor_radius,
            major_radius=major_radius,
            filename=cfg["data_file"],
            exact_radii=True,
        )
    else:
        raise ValueError(f"Unsupported reparameterization: {cfg['parameterization']}")
    dim_x = len(surface.x)
    cfg["num_restarts"] = cfg["num_restarts"] * dim_x
    cfg["raw_samples"] = cfg["raw_samples"] * dim_x
    param_names = [x.replace(",", "_") for x in surface.local_dof_names]
    print(f"dim_x = {dim_x}")
    print(f"param_names = {param_names}")

    # --- Objective ---
    objective = build_objective(cfg, vmec, surface, phiedge_high)

    # --- Bounds ---
    x_min = cfg["x_min"]
    bounds = torch.stack([
        torch.full((dim_x,), x_min, dtype=torch.double, device=device),
        torch.full((dim_x,), 1.0 - x_min, dtype=torch.double, device=device),
    ])
    print(f"Per-DOF bounds: [{x_min}, {1.0 - x_min}]")

    state_file = cfg["state_file"]
    csv_file = cfg["csv_file"]
    num_initial = cfg["num_initial"]
    num_evals = cfg["num_evals"]
    save_frequency = cfg.get("save_frequency", 10)

    X_t = torch.empty((0, dim_x), dtype=torch.double, device=device)
    Y_t = torch.empty((0,), dtype=torch.double, device=device)
    Y_var = torch.empty((0,), dtype=torch.double, device=device)
    Xc_t = None
    C_t = None
    if cfg["use_gp_constraints"]:
        Xc_t = torch.empty((0, dim_x), dtype=torch.double, device=device)
        C_t = torch.empty((0,), dtype=torch.double, device=device)
    results: list[EvalResult] = []
    trial_index = 0
    best_loss_so_far = float("inf")
    penalty = cfg.get("fail_val", 5.5)
    penalty_var = 0.0

    if cfg.get("initial_observations") is not None:
        path = cfg.get("initial_observations")
        X_t = torch.load(f"{path}/X_init.pt", map_location=device)
        Y_t = torch.load(f"{path}/Y_init.pt", map_location=device)
        Y_var = torch.load(f"{path}/Y_var_init.pt", map_location=device)
        if cfg["use_gp_constraints"]:
            Xc_t = torch.load(f"{path}/Xc_init.pt", map_location=device)
            C_t = torch.load(f"{path}/C_init.pt", map_location=device)
        print(f"Initial observations are loaded from {path}")
    else:
        print(f"Running Sobol initial design: {num_initial} point(s).")
        sobol_pool = draw_sobol_samples(
            bounds=bounds, n=num_initial, q=1, seed=cfg["seed"]
        ).squeeze(1)
        while trial_index < num_initial:
            x = sobol_pool[trial_index].cpu().numpy()
            t_eval = time.time()
            result = evaluate(x, objective, vmec, cfg, trial_index, run_name)
            eval_time = time.time() - t_eval
            if not result.failed:
                X_t = torch.cat([X_t, torch.tensor(x, dtype=torch.double, device=device).unsqueeze(0)])
                Y_t = torch.cat([Y_t, torch.tensor([result.loss], dtype=torch.double, device=device)])
                Y_var = torch.cat([Y_var, torch.tensor([result.var_obj], dtype=torch.double, device=device)])
                if cfg["use_gp_constraints"]:
                    Xc_t = torch.cat([Xc_t, torch.tensor(x, dtype=torch.double, device=device).unsqueeze(0)])
                    C_t = torch.cat([C_t, torch.tensor([-1.0], dtype=torch.double, device=device)])
                print(f"Initial observation {trial_index}/{num_initial} done in {eval_time:.1f}s: "
                      f"loss={result.loss:.6e}, worst_dmerc={result.worst_dmerc:.6e}")
            else:
                if cfg["use_gp_constraints"]:
                    Xc_t = torch.cat([Xc_t, torch.tensor(x, dtype=torch.double, device=device).unsqueeze(0)])
                    C_t = torch.cat([C_t, torch.tensor([1.0], dtype=torch.double, device=device)])
                else:
                    X_t = torch.cat([X_t, torch.tensor(x, dtype=torch.double, device=device).unsqueeze(0)])
                    Y_t = torch.cat([Y_t, torch.tensor([penalty], dtype=torch.double, device=device)])
                    Y_var = torch.cat([Y_var, torch.tensor([penalty_var], dtype=torch.double, device=device)])
                print(f"Initial observation {trial_index}/{num_initial} FAILED in {eval_time:.1f}s: "
                f"vmec_failed={result.vmec_failed}, tracing_failed={result.tracing_failed}")
            trial_index += 1
            results.append(result)
            
        save_path = f"alpha_opt/runs/{run_name}"
        os.makedirs(save_path, exist_ok=True)
        torch.save(X_t, f"{save_path}/X_init.pt")
        torch.save(Y_t, f"{save_path}/Y_init.pt")
        torch.save(Y_var, f"{save_path}/Y_var_init.pt")
        if cfg["use_gp_constraints"]:
            torch.save(Xc_t, f"{save_path}/Xc_init.pt")
            torch.save(C_t, f"{save_path}/C_init.pt")
        try:
            artifact = wandb.Artifact("initial_observations", type="dataset")
            artifact.add_file(f"{save_path}/X_init.pt")
            artifact.add_file(f"{save_path}/Y_init.pt")
            artifact.add_file(f"{save_path}/Y_var_init.pt")
            if cfg["use_gp_constraints"]:
                artifact.add_file(f"{save_path}/Xc_init.pt")
                artifact.add_file(f"{save_path}/C_init.pt")
            wandb_run.log_artifact(artifact)
            print("Uploaded initial observations to wandb.")
        except Exception as e:
            print(f"Warning: failed to upload initial observations artifact: {e}")
        print(f"Saved initial observations to {save_path}/")

    # --- Bayesian optimisation loop ---
    for i in range(trial_index, num_evals):
        t_gen = time.time()
        acq_val = None
        x, acq_val, gp_hparams, candidate_variance = propose_next(X_t, Y_t, Y_var, bounds, cfg, Xc_t, C_t)
        gen_time = time.time() - t_gen
        print(f"Iteration {i}: candidate generated in {gen_time:.2f}s, "
                f"acq={acq_val:.4e}")

        t_eval = time.time()
        result = evaluate(x.cpu().numpy(), objective, vmec, cfg, i, run_name)
        eval_time = time.time() - t_eval

        if not result.failed:
            X_t = torch.cat([X_t, x.unsqueeze(0)])
            Y_t = torch.cat([Y_t, torch.tensor([result.loss], dtype=torch.double, device=device)])
            Y_var = torch.cat([Y_var, torch.tensor([result.var_obj], dtype=torch.double, device=device)])
            if result.loss < best_loss_so_far:
                best_loss_so_far = result.loss
        else:
            if cfg["use_gp_constraints"]:
                Xc_t = torch.cat([Xc_t, x.unsqueeze(0)])
                C_t = torch.cat([C_t, torch.tensor([1.0], dtype=torch.double, device=device)])
                result.loss = None
            else:
                X_t = torch.cat([X_t, x.unsqueeze(0)])
                Y_t = torch.cat([Y_t, torch.tensor([penalty], dtype=torch.double, device=device)])
                Y_var = torch.cat([Y_var, torch.tensor([penalty_var], dtype=torch.double, device=device)])
                result.loss = penalty

        # Z-score of the new observation against the sample mean/variance of
        # all prior observations (excludes the point just appended above and
        # any prior penalty/failed points).
        prior_y = Y_t[:-1]
        prior_y = prior_y[prior_y != penalty]
        if prior_y.numel() >= 2:
            prior_mean = prior_y.mean().item()
            prior_std = prior_y.std().item()
            z_score = (Y_t[-1].item() - prior_mean) / prior_std if prior_std > 0 else float("nan")
        else:
            z_score = float("nan")

        wandb.log(
            {
                "iteration": i,
                "loss": result.loss,
                "worst_dmerc": result.worst_dmerc,
                "vmec_failed": (result.vmec_failed)*1.0,
                "tracing_failed": (result.tracing_failed)*1.0,
                "case": result.case if result.case is not None else -1,
                "var_obj": result.var_obj if result.var_obj is not None else float("nan"),
                "best_loss": best_loss_so_far,
                "acq_value": acq_val if acq_val is not None else float("nan"),
                "candidate_variance": candidate_variance,
                "z_score": z_score,
                "gen_time_s": gen_time,
                "eval_time_s": eval_time,
                **{f"x/{name}": float(val) for name, val in zip(param_names, x.cpu().tolist())},
                **gp_hparams,
            },
            step=i,
        )
        results.append(result)
        if i % save_frequency == 0:
            # --- Final save ---
            X_np = X_t.cpu().numpy()
            Y_np = Y_t.cpu().numpy()
            Y_var_np = Y_var.cpu().numpy()
            if not cfg["use_gp_constraints"]:
                save_state(state_file, csv_file, param_names, X_np, Y_np, Y_var_np, results)
            else:
                Xc_np = Xc_t.cpu().numpy()
                C_np = C_t.cpu().numpy()
                save_state_with_c(state_file, csv_file, param_names, X_np, Y_np, Y_var_np, Xc_np, C_np, results)
            try:
                artifact = wandb.Artifact(f"state_{len(X_np):06d}", type="checkpoint")
                for path in [state_file, csv_file]:
                    if os.path.exists(path):
                        artifact.add_file(path)
                wandb_run.log_artifact(artifact)
            except Exception as e:
                print(f"Warning: failed to upload state artifact: {e}")
            print(f"Wrote {state_file} and {csv_file}.")

    # --- Final save ---
    X_np = X_t.cpu().numpy()
    Y_np = Y_t.cpu().numpy()
    Y_var_np = Y_var.cpu().numpy()
    save_state(state_file, csv_file, param_names, X_np, Y_np, Y_var_np, results)
    try:
        artifact = wandb.Artifact(f"state_{len(X_np):06d}", type="checkpoint")
        for path in [state_file, csv_file]:
            if os.path.exists(path):
                artifact.add_file(path)
        wandb_run.log_artifact(artifact)
    except Exception as e:
        print(f"Warning: failed to upload state artifact: {e}")
    print(f"Wrote {state_file} and {csv_file}.")

    wandb.finish()


if __name__ == "__main__":
    main()