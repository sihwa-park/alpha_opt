#!/usr/bin/env python
"""Compute the exact per-dimension whitened partial derivatives
v_i = lengthscale_i * dg/dx_i at x_best, one coordinate-direction central
difference per interior DOF.

`wandb_best_x_directional_cd.py` only gives random-direction projections
z^T v (Monte Carlo over `--n-random-directions` directions) and needs a
moment-based correction for bound-clipping to say anything about
||D grad g|| (see `notebooks/directional_cd_error_analysis.ipynb`). This
script instead isolates each v_i directly and exactly (up to truncation +
objective noise -- no random-projection aggregation, no isotropy
assumption), by stepping along one standard basis direction at a time:

    v_i = (g(x_best + h_i e_i) - g(x_best - h_i e_i)) / (2 * perturb_scale),
    h_i = perturb_scale * lengthscale_i (symmetrically clipped to stay in
    bounds, same convention as wandb_best_x_directional_cd.py).

DOFs pinned exactly at the box edge are skipped -- a central difference is
impossible there (one side of any step is infeasible). Those DOFs get their
own one-sided diagnostic in `wandb_boundary_activation_check.py`; this
script does not need or use that output, since the goal here is the
individual v_i values, not an aggregate norm.

Evaluation budget for `garabedian_linear_cei_pressure`'s current x_best
(23 DOFs, 8 pinned at the box edge, 15 interior): 2 * 15 = 30 real
evaluations, one central difference per interior DOF.

Checkpoints to `<output-dir>/coordinate_cd.csv` after every dimension;
resumable via the same --output-dir.

Usage:
    python wandb_exact_gradient_coordinate_cd.py --project garabedian_linear_cei_pressure
"""

import argparse
import csv
import json
import os
import time

import pandas as pd
import wandb
import yaml

from wandb_best_x_directional_cd import (
    DEFAULT_ENTITY,
    append_result_row,
    build_real_objective,
    evaluate_direction,
    find_global_best,
    get_x_best_and_lengthscale,
)
from wandb_boundary_activation_check import find_boundary_dims


# ---------------------------------------------------------------------------
# Interior-dim step + clip (mirrors sample_direction_step's symmetric clip,
# specialized to a single coordinate rather than a random full-dim vector)
# ---------------------------------------------------------------------------

def coordinate_step(x_best_i, x_min, lengthscale_i, perturb_scale):
    half_width = max(min(x_best_i - x_min, (1.0 - x_min) - x_best_i), 0.0)
    h_raw = perturb_scale * lengthscale_i
    h = min(h_raw, half_width)
    return h, h != h_raw  # (clipped step magnitude, whether it was clipped)


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------

def load_completed_dims(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        return {int(row["dim"]) for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="garabedian_linear_cei_pressure", help="wandb project name")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="wandb entity (team/user)")
    parser.add_argument("--config-path", default="../configs/garabedian_linear_pressure.yaml",
                         help="local yaml config matching this project's runs (bounds, VMEC input, kernel)")
    parser.add_argument("--perturb-scale", type=float, default=0.05,
                         help="step size constant c; each step is h_i = c * lengthscale_i")
    parser.add_argument("--output-dir", default="exact_gradient_results",
                         help="where to write the checkpointed CSV and summary")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "coordinate_cd.csv")
    eval_root = os.path.join(args.output_dir, "evals")

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("wandb_project") != args.project:
        print(f"WARNING: --config-path's wandb_project ('{cfg.get('wandb_project')}') "
              f"does not match --project ('{args.project}'). Double check --config-path.")

    api = wandb.Api()
    best_run, best_row, best_df, i_star = find_global_best(api, args.entity, args.project)
    x_best, param_names, lengthscale_logged, y_best = get_x_best_and_lengthscale(
        best_run, best_row, i_star, cfg["use_gp_constraints"],
    )

    objective, dim = build_real_objective(cfg)
    if dim != len(x_best):
        raise RuntimeError(
            f"Surface built from --config-path has dim={dim}, but x_best from wandb/checkpoint "
            f"has dim={len(x_best)}. --config-path likely doesn't match this project's runs."
        )
    x_min = float(cfg["x_min"])

    boundary_dims = {i for i, _side, _sign in find_boundary_dims(x_best, x_min)}
    interior_dims = [i for i in range(dim) if i not in boundary_dims]
    print(f"\n{len(boundary_dims)}/{dim} DOFs boundary-pinned (skipped here -- see "
          f"wandb_boundary_activation_check.py), {len(interior_dims)}/{dim} interior "
          f"(central differences computed here).")

    # --- Interior DOFs: fresh central differences, checkpointed ---
    fieldnames = ["dim", "param_name", "step", "clipped", "loss_plus", "loss_minus",
                  "case_p", "var_obj_p", "case_m", "var_obj_m", "v_i",
                  "vmec_failed", "tracing_failed", "eval_time_s"]
    done = load_completed_dims(csv_path)
    if done:
        print(f"\nResuming: {len(done)} interior dim(s) already evaluated in {csv_path}")

    for i in interior_dims:
        if i in done:
            continue
        h, clipped = coordinate_step(x_best[i], x_min, lengthscale_logged[i], args.perturb_scale)
        x_plus, x_minus = x_best.copy(), x_best.copy()
        x_plus[i] += h
        x_minus[i] -= h
        eval_dir_plus = os.path.join(eval_root, f"{best_run.name}_dim{i:02d}_{param_names[i]}_plus")
        eval_dir_minus = os.path.join(eval_root, f"{best_run.name}_dim{i:02d}_{param_names[i]}_minus")

        t0 = time.time()
        loss_plus, case_p, var_obj_p, vmec_failed_p, tracing_failed_p = evaluate_direction(
            objective, x_plus, eval_dir_plus
        )
        loss_minus, case_m, var_obj_m, vmec_failed_m, tracing_failed_m = evaluate_direction(
            objective, x_minus, eval_dir_minus
        )
        eval_time = time.time() - t0
        vmec_failed = vmec_failed_p or vmec_failed_m
        tracing_failed = tracing_failed_p or tracing_failed_m

        if loss_plus is not None and loss_minus is not None and h > 0:
            v_i = (loss_plus - loss_minus) / (2.0 * h)
        else:
            v_i = None

        row = {
            "dim": i, "param_name": param_names[i], "step": h, "clipped": int(clipped),
            "loss_plus": loss_plus, "loss_minus": loss_minus,
            "case_p": case_p, "var_obj_p": var_obj_p, "case_m": case_m, "var_obj_m": var_obj_m,
            "v_i": v_i, "vmec_failed": int(vmec_failed), "tracing_failed": int(tracing_failed),
            "eval_time_s": eval_time,
        }
        append_result_row(csv_path, row, fieldnames)
        print(f"[dim {i:02d} {param_names[i]}] step={h:.4e} (clipped={clipped})  "
              f"loss_plus={loss_plus}  loss_minus={loss_minus}  v_i={v_i}  ({eval_time:.1f}s)")

    # --- Report the per-dimension v_i values (interior DOFs only) ---
    interior_df = pd.read_csv(csv_path)
    v_by_dim = {}
    n_failed = 0
    for _, r in interior_df.iterrows():
        i = int(r["dim"])
        if pd.notna(r["v_i"]):
            v_by_dim[i] = float(r["v_i"])
        else:
            n_failed += 1

    print(f"\nv_i computed for {len(v_by_dim)}/{len(interior_dims)} interior DOFs "
          f"({len(boundary_dims)} boundary DOFs excluded by design -- see "
          "wandb_boundary_activation_check.py for those):")
    for i in interior_dims:
        val = v_by_dim.get(i)
        print(f"  v[{i:02d}] {param_names[i]:>6} = {val:+.4f}" if val is not None
              else f"  v[{i:02d}] {param_names[i]:>6} = FAILED")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "run": best_run.name, "iteration": i_star, "param_names": param_names,
            "v_by_dim": {param_names[i]: val for i, val in v_by_dim.items()},
            "n_failed": n_failed,
            "boundary_dims": sorted(boundary_dims), "interior_dims": interior_dims,
            "perturb_scale": args.perturb_scale,
        }, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
