#!/usr/bin/env python
"""Real-objective check of whether the box boundary is an ACTIVE constraint
at x_best, for every DOF where x_best sits exactly on the box edge.

The global best `x_best` found in `garabedian_linear_cei_pressure`
(iteration 779 of run 'fanciful-sweep-10', as characterized in
`notebooks/directional_cd_error_analysis.ipynb`) has 8 of its 23 DOFs pinned
exactly at `x_min` or `1 - x_min`. A central difference is impossible there
(one side of any step leaves the feasible box), so this script does the only
thing that IS feasible: evaluate the real objective at a single step INTO
the box along that one coordinate, and compare to the loss at x_best itself.

Interpretation (we are MINIMIZING `loss`):
  - loss increases moving into the box  -> the boundary is genuinely
    ACTIVE: the unconstrained gradient wants to push further outward
    (infeasible), so staying pinned at the edge is locally optimal.
  - loss decreases moving into the box  -> the boundary is NOT active in
    any KKT sense: moving inward improves the objective, so x_best is not
    yet locally optimal along that coordinate. This points at `x_min` being
    too restrictive (the true optimum lies further inside than the box
    forces it to explore FROM, i.e. the search got stuck at the edge) or
    inadequate local refinement near the boundary, per the project skill's
    "Known findings".

Both directions of evidence require a noise-aware significance test, since
the objective is itself a noisy Monte-Carlo estimate (delta-method
`var_obj`, real for case 2, a hardcoded placeholder for case 1 -- flagged
per-dimension if this comes up).

This is step 1 of a two-script pair: `wandb_exact_gradient_coordinate_cd.py`
reuses this script's checkpointed results for the same 8 DOFs (no duplicate
real evaluations) plus 2*(dim - 8) fresh central-difference evaluations on
the remaining interior DOFs, to reconstruct the FULL whitened gradient
vector D grad(g), exactly, component by component.

Like the other directional scripts, this runs sequentially (one real
VMEC + tracing evaluation per pinned DOF, plus one shared reference
evaluation at x_best), checkpoints after every unit of work, and is safe to
resume by rerunning with the same --output-dir.

Usage:
    python wandb_boundary_activation_check.py --project garabedian_linear_cei_pressure
"""

import argparse
import csv
import json
import os
import time

import numpy as np
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

BOUNDARY_TOL = 1e-9


# ---------------------------------------------------------------------------
# Boundary-dim detection
# ---------------------------------------------------------------------------

def find_boundary_dims(x_best, x_min):
    """Return a list of (dim_idx, pinned_at, sign) for every DOF pinned
    exactly at x_min or 1 - x_min. `sign` is the direction that moves INTO
    the feasible box (+1 if pinned at x_min, -1 if pinned at 1 - x_min)."""
    boundary_dims = []
    for i, xi in enumerate(x_best):
        if abs(xi - x_min) <= BOUNDARY_TOL:
            boundary_dims.append((i, "x_min", +1))
        elif abs(xi - (1.0 - x_min)) <= BOUNDARY_TOL:
            boundary_dims.append((i, "1-x_min", -1))
    return boundary_dims


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------

def load_xbest_reference(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


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
                         help="step size constant c; each boundary step is h_i = c * lengthscale_i")
    parser.add_argument("--z-sig", type=float, default=2.0,
                         help="|z_score| threshold for calling a direction's sign statistically significant")
    parser.add_argument("--output-dir", default="boundary_activation_results",
                         help="where to write the checkpointed CSV, reference JSON, and summary")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "boundary_activation.csv")
    ref_path = os.path.join(args.output_dir, "xbest_reference.json")
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

    boundary_dims = find_boundary_dims(x_best, x_min)
    print(f"\n{len(boundary_dims)}/{dim} DOFs pinned exactly at the box edge: "
          f"{[(param_names[i], side) for i, side, _ in boundary_dims]}")
    if not boundary_dims:
        print("No boundary-pinned DOFs found for this x_best -- nothing to check.")
        return

    # --- Shared reference evaluation at x_best itself (once, resumable) ---
    ref = load_xbest_reference(ref_path)
    if ref is not None:
        print(f"\nReusing cached x_best reference eval: loss={ref['loss']:.6e} case={ref['case']}")
    else:
        print("\nEvaluating the real objective at x_best (shared reference for every boundary dim)...")
        t0 = time.time()
        loss_xbest, case_xbest, var_obj_xbest, vmec_failed, tracing_failed = evaluate_direction(
            objective, x_best, os.path.join(eval_root, f"{best_run.name}_xbest_ref")
        )
        eval_time = time.time() - t0
        print(f"objective(x_best) = {loss_xbest}  (wandb-logged: {y_best:.6e})  case={case_xbest}  "
              f"var_obj={var_obj_xbest}  [{eval_time:.1f}s, vmec_failed={vmec_failed}, "
              f"tracing_failed={tracing_failed}]")
        if loss_xbest is not None and not np.isclose(loss_xbest, y_best, rtol=1e-3, atol=1e-6):
            print("WARNING: objective(x_best) does not match the wandb-logged loss within tolerance -- "
                  "the reconstructed VMEC/surface/config setup may not exactly match the original run.")
        ref = {
            "run": best_run.name, "iteration": i_star, "x_best": x_best.tolist(),
            "param_names": param_names, "y_best_logged": y_best,
            "loss": loss_xbest, "case": case_xbest, "var_obj": var_obj_xbest,
            "vmec_failed": vmec_failed, "tracing_failed": tracing_failed, "eval_time_s": eval_time,
        }
        with open(ref_path, "w") as f:
            json.dump(ref, f, indent=2)

    if ref["loss"] is None or ref["vmec_failed"] or ref["tracing_failed"]:
        raise RuntimeError("x_best reference evaluation failed -- cannot compare boundary steps against it. "
                            f"Delete {ref_path} to retry.")

    # --- Per-boundary-dim one-sided step into the box ---
    fieldnames = ["dim", "param_name", "pinned_at", "x_best_i", "lengthscale_i", "step",
                  "loss_perturbed", "case_p", "var_obj_p", "delta", "z_score", "verdict",
                  "vmec_failed", "tracing_failed", "eval_time_s"]
    done = load_completed_dims(csv_path)
    if done:
        print(f"\nResuming: {len(done)} boundary dim(s) already evaluated in {csv_path}")

    for i, pinned_at, sign in boundary_dims:
        if i in done:
            continue
        step = sign * args.perturb_scale * lengthscale_logged[i]
        x_trial = x_best.copy()
        x_trial[i] += step
        eval_dir = os.path.join(eval_root, f"{best_run.name}_dim{i:02d}_{param_names[i]}")

        t0 = time.time()
        loss_p, case_p, var_obj_p, vmec_failed, tracing_failed = evaluate_direction(objective, x_trial, eval_dir)
        eval_time = time.time() - t0

        if loss_p is None:
            delta, z_score, verdict = None, None, "eval_failed"
        else:
            delta = loss_p - ref["loss"]
            if (ref["var_obj"] is not None and var_obj_p is not None
                    and ref["case"] == 2 and case_p == 2):
                z_score = delta / np.sqrt(ref["var_obj"] + var_obj_p)
            else:
                z_score = None  # at least one side is case 1 -> var_obj is a placeholder, not a real SE
            if z_score is not None and abs(z_score) >= args.z_sig:
                verdict = "active (boundary optimal)" if delta > 0 else "inactive (interior improves)"
            else:
                sign_word = "loss up" if (delta is not None and delta > 0) else "loss down"
                verdict = f"inconclusive ({sign_word}, not significant or var_obj unreliable)"

        row = {
            "dim": i, "param_name": param_names[i], "pinned_at": pinned_at,
            "x_best_i": float(x_best[i]), "lengthscale_i": float(lengthscale_logged[i]), "step": step,
            "loss_perturbed": loss_p, "case_p": case_p, "var_obj_p": var_obj_p,
            "delta": delta, "z_score": z_score, "verdict": verdict,
            "vmec_failed": int(vmec_failed), "tracing_failed": int(tracing_failed), "eval_time_s": eval_time,
        }
        append_result_row(csv_path, row, fieldnames)
        print(f"[dim {i:02d} {param_names[i]}, pinned at {pinned_at}] step={step:+.4e}  "
              f"loss_perturbed={loss_p}  delta={delta}  z={z_score}  -> {verdict}  "
              f"({eval_time:.1f}s)")

    # --- Summary ---
    results_df = pd.read_csv(csv_path)
    n_active = (results_df["verdict"] == "active (boundary optimal)").sum()
    n_inactive = (results_df["verdict"] == "inactive (interior improves)").sum()
    n_inconclusive = len(results_df) - n_active - n_inactive
    print(f"\n{len(results_df)}/{len(boundary_dims)} boundary dims checked: "
          f"{n_active} active, {n_inactive} inactive (interior improves), {n_inconclusive} inconclusive.")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "run": best_run.name, "iteration": i_star,
            "n_boundary_dims": len(boundary_dims), "n_active": int(n_active),
            "n_inactive": int(n_inactive), "n_inconclusive": int(n_inconclusive),
            "perturb_scale": args.perturb_scale, "z_sig_threshold": args.z_sig,
        }, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
