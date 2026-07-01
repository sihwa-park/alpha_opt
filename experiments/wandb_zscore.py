#!/usr/bin/env python
"""
Fit a GP (matching the one in botorch_optimize.py) to the design
parameters/objective values of one wandb run ("run A"), then use the GP's
predictive mean/variance at run B's design parameters to print the Z-score of
every value in an index range of a second wandb run ("run B").

Usage:
    python wandb_zscore.py --project PROJECT \
        --run-a RUN_A_NAME --range-a START END \
        --run-b RUN_B_NAME --range-b START END

Example:
    python wandb_zscore.py --project garabedian_linear_cei \
        --run-a divine-firebrand-74 --range-a 0 50 \
        --run-b solar-comet-12     --range-b 0 50
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.mlls import ExactMarginalLogLikelihood

DEFAULT_ENTITY = "sp2582-cornell-university"


def fetch_history(api, project, entity, run_name, index_key, metric, param_prefix, var_obj_key="var_obj"):
    """Return sorted (index, metric, params, var_obj) arrays for a wandb run, by display name."""
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"display_name": run_name})
    if not runs:
        raise ValueError(f"No wandb run found with name '{run_name}' in '{path}'")
    run = runs[0]
    history = list(run.scan_history())
    rows = [row for row in history if index_key in row and metric in row]
    if not rows:
        raise ValueError(f"Run '{run_name}' has no logged '{metric}'/'{index_key}'")
    order = np.argsort([row[index_key] for row in rows])
    indices = np.array([rows[i][index_key] for i in order])
    values = np.array([rows[i][metric] for i in order], dtype=float)
    param_keys = sorted({k for row in rows for k in row if k.startswith(param_prefix)})
    if not param_keys:
        raise ValueError(f"No '{param_prefix}*' parameter columns found for run '{run_name}'")
    params = np.array([[rows[i].get(k, np.nan) for k in param_keys] for i in order], dtype=float)
    var_obj = None
    if all(var_obj_key in rows[i] for i in order):
        var_obj = np.array([rows[i][var_obj_key] for i in order], dtype=float)
    return indices, values, params, param_keys, var_obj


def slice_range(indices, values, params, index_range, var_obj=None):
    lo, hi = index_range
    mask = (indices >= lo) & (indices <= hi)
    sliced_var = var_obj[mask] if var_obj is not None else None
    return indices[mask], values[mask], params[mask], sliced_var


def fit_gp(train_X, train_Y, train_Yvar, bounds):
    """Fit a SingleTaskGP on (params, objective), matching propose_next() in botorch_optimize.py."""
    model = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        covar_module=get_covar_module_with_dim_scaled_prior(ard_num_dims=train_X.shape[-1]),
        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="wandb entity (team/user)")
    parser.add_argument("--run-a", required=True, help="run name to fit the GP on")
    parser.add_argument("--range-a", nargs=2, type=int, metavar=("START", "END"), required=True,
                         help="inclusive index range in run A used to fit the GP")
    parser.add_argument("--run-b", required=True, help="run name to compute Z-scores for")
    parser.add_argument("--range-b", nargs=2, type=int, metavar=("START", "END"), required=True,
                         help="inclusive index range in run B to report Z-scores for")
    parser.add_argument("--metric", default="loss", help="objective metric key logged to wandb")
    parser.add_argument("--index-key", default="iteration", help="step/index key logged to wandb")
    parser.add_argument("--param-prefix", default="x/",
                         help="prefix of the design-parameter keys logged to wandb (e.g. 'x/name')")
    parser.add_argument("--var-obj-key", default="var_obj",
                         help="key logged for the objective's observation noise variance (used as train_Yvar)")
    parser.add_argument("--plot-output", default="zscore_plot.png",
                         help="path to save the Z-score plot for run B")
    args = parser.parse_args()

    api = wandb.Api()

    a_idx, a_val, a_params, param_keys, a_var = fetch_history(
        api, args.project, args.entity, args.run_a, args.index_key, args.metric,
        args.param_prefix, args.var_obj_key,
    )
    a_idx, a_val, a_params, a_var = slice_range(a_idx, a_val, a_params, args.range_a, a_var)
    finite_mask = np.isfinite(a_val) & np.isfinite(a_params).all(axis=1)
    a_val, a_params = a_val[finite_mask], a_params[finite_mask]
    a_var = a_var[finite_mask] if a_var is not None else None
    if a_val.size < 2:
        raise ValueError(f"Run A range {args.range_a} has fewer than 2 finite '{args.metric}' values")
    print(f"Run A '{args.run_a}' [{args.range_a[0]}, {args.range_a[1]}]: "
          f"n={a_val.size}, params={param_keys}")

    b_idx, b_val, b_params, b_param_keys, b_var = fetch_history(
        api, args.project, args.entity, args.run_b, args.index_key, args.metric,
        args.param_prefix, args.var_obj_key,
    )
    if b_param_keys != param_keys:
        raise ValueError(f"Run A params {param_keys} != run B params {b_param_keys}")
    b_idx, b_val, b_params, b_var = slice_range(b_idx, b_val, b_params, args.range_b, b_var)

    bounds_np = np.stack([
        np.minimum(a_params.min(axis=0), np.nanmin(b_params, axis=0)),
        np.maximum(a_params.max(axis=0), np.nanmax(b_params, axis=0)),
    ])
    bounds_np[1] = np.where(bounds_np[1] > bounds_np[0], bounds_np[1], bounds_np[0] + 1.0)
    bounds = torch.tensor(bounds_np, dtype=torch.double)

    train_X = torch.tensor(a_params, dtype=torch.double)
    train_Y = torch.tensor(a_val, dtype=torch.double).unsqueeze(-1)
    train_Yvar = torch.tensor(a_var, dtype=torch.double).unsqueeze(-1) if a_var is not None else None
    model = fit_gp(train_X, train_Y, train_Yvar, bounds)

    test_mask = np.isfinite(b_val) & np.isfinite(b_params).all(axis=1)
    test_X = torch.tensor(b_params[test_mask], dtype=torch.double)
    with torch.no_grad():
        posterior = model.posterior(test_X)
        pred_mean_obs = posterior.mean.squeeze(-1).numpy()
        pred_var_obs = posterior.variance.squeeze(-1).numpy()

    pred_mean = np.full(b_val.shape, np.nan)
    pred_var = np.full(b_val.shape, np.nan)
    pred_mean[test_mask] = pred_mean_obs
    pred_var[test_mask] = pred_var_obs

    z_scores = np.where(np.isfinite(b_val) & (pred_var > 0), (b_val - pred_mean) / np.sqrt(pred_var), np.nan)

    print(f"\nZ-scores for run B '{args.run_b}' [{args.range_b[0]}, {args.range_b[1]}] "
          f"(GP fit on run A '{args.run_a}'):")
    for idx, val, mean, var, z in zip(b_idx, b_val, pred_mean, pred_var, z_scores):
        print(f"  {args.index_key}={idx:.0f}  {args.metric}={val:.6e}  "
              f"gp_mean={mean:.6e}  gp_std={var ** 0.5:.6e}  z_score={z:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(b_idx, z_scores, marker="o", linestyle="-")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(args.index_key)
    ax.set_ylabel("Z-score")
    ax.set_title(f"Z-score of '{args.metric}' for run B '{args.run_b}'\n"
                 f"(GP predictive mean/std fit on run A '{args.run_a}' [{args.range_a[0]}, {args.range_a[1]}])")
    fig.tight_layout()
    fig.savefig(args.plot_output, dpi=150)
    print(f"\nSaved plot to {args.plot_output}")


if __name__ == "__main__":
    main()
