#!/usr/bin/env python
"""Real-objective directional finite differences + reseed noise check around
points near the case-1/case-2 boundary of the alpha-loss objective.

`alpha_opt/tracing/loss_times.py`'s `alpha_loss_objective_from_times` splits
into two regimes depending on whether the energy-loss threshold is crossed
before `t_max`:
    case 1: crossed early  -> objective = -log10(t_exceed), var_obj = 1e-5 (constant)
    case 2: never crossed  -> objective from the energy-loss fraction, var_obj
                               computed analytically (variance_case2)
The boundary is t_exceed -> t_max, i.e. loss -> -log10(t_max).

This script:
  1. Finds the `--n-boundary-points` wandb trials (pooled across every run in
     the project) whose logged loss is closest to that boundary.
  2. For each point, re-evaluates the real objective at x to cross-check the
     reconstructed setup and learn its case (this also produces `wout_tmp.nc`
     in that point's eval dir).
  3. If the point comes back as case 1 -- where var_obj is a hardcoded
     constant rather than a measured quantity -- re-traces the SAME VMEC
     equilibrium with two different particle-sampling seeds (no VMEC rerun)
     to empirically see the noise that constant is standing in for.
  4. Runs the same length-scale-weighted central-difference directional
     gradient diagnostic as wandb_best_x_directional_cd.py, but with only
     `--n-random-directions` (default 2) directions per point.

Like wandb_best_x_directional_cd.py, this runs sequentially, can take a long
time, and checkpoints to `<output-dir>/*.csv` after every unit of work so an
interrupted run can be resumed by rerunning with the same --output-dir.

Usage:
    python wandb_boundary_seed_directional_cd.py --project garabedian_linear_cei_pressure
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import wandb
import yaml

from alpha_opt.tracing import compute_alpha_loss
from wandb_best_x_directional_cd import (
    DEFAULT_ENTITY,
    append_result_row,
    build_real_objective,
    evaluate_direction,
    fetch_run_dataframe,
    get_x_best_and_lengthscale,
    sample_direction_step,
)


# ---------------------------------------------------------------------------
# Step 1: find the points closest to the case-1/case-2 boundary
# ---------------------------------------------------------------------------

def find_near_boundary(api, entity, project, loss_boundary, n_points):
    """Pool every valid trial across every run in the project, rank by
    distance to `loss_boundary`, and return the closest `n_points` as
    (dist, run, row, df, iteration) tuples."""
    path = f"{entity}/{project}" if entity else project
    runs = list(api.runs(path))
    print(f"Found {len(runs)} run(s) in '{path}'")

    candidates = []
    for run in runs:
        df = fetch_run_dataframe(run)
        if df is None:
            continue
        valid = df[(df["vmec_failed"] == 0) & (df["tracing_failed"] == 0) & np.isfinite(df["loss"])]
        for _, row in valid.iterrows():
            dist = abs(float(row["loss"]) - loss_boundary)
            candidates.append((dist, run, row, df, int(row["iteration"])))

    if not candidates:
        raise RuntimeError(f"No run in '{path}' has any valid logged 'loss'.")

    candidates.sort(key=lambda c: c[0])
    chosen = candidates[:n_points]
    print(f"Selected {len(chosen)} point(s) closest to boundary loss={loss_boundary:.6f}:")
    for i, (dist, run, row, _df, i_star) in enumerate(chosen):
        print(f"  [{i}] run='{run.name}' iteration={i_star} loss={row['loss']:.6e} dist={dist:.6e}")
    return chosen


# ---------------------------------------------------------------------------
# Step 2: reseed comparison (case-1 points only) -- re-trace only, no VMEC
# ---------------------------------------------------------------------------

def evaluate_reseed(cfg, eval_dir, seed):
    """Re-trace the wout_tmp.nc already sitting in `eval_dir` (written by a
    prior evaluate_direction call at that point's x) with a different
    particle-sampling seed. Does NOT re-run VMEC -- same equilibrium, only
    the tracing initial conditions change."""
    cwd = os.getcwd()
    os.chdir(eval_dir)
    try:
        loss, case, var_obj = compute_alpha_loss(
            "wout_tmp.nc",
            n_particles=cfg["n_particles"],
            t_max=cfg["t_max"],
            tau=cfg["tau"],
            t_block=cfg["t_block"],
            min_dt=cfg["min_dt"],
            maxloss=cfg["maxloss"],
            tol=cfg["tol"],
            vacuum=cfg["vacuum"],
            seed=seed,
        )
        return float(loss), case, var_obj, False
    except Exception as e:
        print(f"Reseed re-trace (seed={seed}) failed: {e}")
        return None, None, None, True
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------

def load_completed_points(csv_path):
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path) as f:
        return {int(row["point"]): row for row in csv.DictReader(f)}


def load_completed_reseed_points(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        return {int(row["point"]) for row in csv.DictReader(f)}


def load_completed_point_directions(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        return {(int(row["point"]), int(row["direction"])) for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="garabedian_linear_cei_pressure", help="wandb project name")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="wandb entity (team/user)")
    parser.add_argument("--config-path", default="configs/garabedian_linear_pressure.yaml",
                         help="local yaml config matching this project's runs (bounds, VMEC input, kernel)")
    parser.add_argument("--perturb-scale", type=float, default=0.05,
                         help="step size constant c; each directional step is h = c * lengthscale * z")
    parser.add_argument("--n-boundary-points", type=int, default=10,
                         help="number of near-boundary points to evaluate, pooled across all runs")
    parser.add_argument("--n-random-directions", type=int, default=2,
                         help="number of random lengthscale-weighted directions to evaluate PER POINT")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="boundary_seed_cd_results",
                         help="where to write the checkpointed CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    points_csv = os.path.join(args.output_dir, "boundary_points.csv")
    reseed_csv = os.path.join(args.output_dir, "reseed_comparison.csv")
    directional_csv = os.path.join(args.output_dir, "directional_cd.csv")
    eval_root = os.path.join(args.output_dir, "evals")

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("wandb_project") != args.project:
        print(f"WARNING: --config-path's wandb_project ('{cfg.get('wandb_project')}') "
              f"does not match --project ('{args.project}'). Double check --config-path.")

    loss_boundary = -float(np.log10(cfg["t_max"]))
    print(f"Case 1 / case 2 boundary loss = -log10(t_max) = -log10({cfg['t_max']}) = {loss_boundary:.6f}")

    api = wandb.Api()
    boundary_points = find_near_boundary(api, args.entity, args.project, loss_boundary, args.n_boundary_points)

    objective, dim = build_real_objective(cfg)
    x_min = float(cfg["x_min"])

    points_fieldnames = ["point", "run", "iteration", "loss_logged", "dist_to_boundary",
                          "loss_eval", "case", "var_obj", "vmec_failed", "tracing_failed", "eval_time_s"]
    reseed_fieldnames = ["point", "seed_a", "seed_b", "loss_a", "loss_b", "diff",
                          "case_a", "case_b", "tracing_failed_a", "tracing_failed_b"]
    directional_fieldnames = ["point", "direction", "step_norm", "loss_plus", "loss_minus", "gradient",
                               "case_p", "var_obj_p", "case_m", "var_obj_m", "clipped",
                               "vmec_failed", "tracing_failed", "eval_time_s", "h_json", "z"]

    completed_points = load_completed_points(points_csv)
    completed_reseed = load_completed_reseed_points(reseed_csv)
    completed_directions = load_completed_point_directions(directional_csv)
    if completed_points:
        print(f"\nResuming: {len(completed_points)} point(s) already evaluated in {points_csv}")

    for p_idx, (dist, run, row, _df, i_star) in enumerate(boundary_points):
        eval_dir_base = os.path.join(eval_root, f"pt{p_idx:02d}_{run.name}_base")

        x_p, param_names, lengthscale_p, y_logged = get_x_best_and_lengthscale(
            run, row, i_star, cfg["use_gp_constraints"],
        )
        if len(lengthscale_p) != dim:
            raise RuntimeError(
                f"Point {p_idx}: lengthscale dim ({len(lengthscale_p)}) != surface dim ({dim}); "
                "--config-path likely doesn't match this project's runs."
            )

        if p_idx in completed_points:
            prow = completed_points[p_idx]
            case = int(prow["case"]) if prow["case"] not in ("", "None") else None
            vmec_failed = bool(int(prow["vmec_failed"]))
            tracing_failed = bool(int(prow["tracing_failed"]))
            print(f"\n[point {p_idx}] already evaluated (loss_eval={prow['loss_eval']}, case={case})")
        else:
            print(f"\n[point {p_idx}] run='{run.name}' iteration={i_star} "
                  f"loss_logged={row['loss']:.6e} dist_to_boundary={dist:.6e}")
            t0 = time.time()
            loss_eval, case, var_obj, vmec_failed, tracing_failed = evaluate_direction(
                objective, x_p, eval_dir_base
            )
            eval_time = time.time() - t0
            print(f"[point {p_idx}] objective(x) = {loss_eval}  case={case}  var_obj={var_obj}  "
                  f"[{eval_time:.1f}s, vmec_failed={vmec_failed}, tracing_failed={tracing_failed}]")
            append_result_row(points_csv, {
                "point": p_idx, "run": run.name, "iteration": i_star,
                "loss_logged": float(row["loss"]), "dist_to_boundary": dist,
                "loss_eval": loss_eval, "case": case, "var_obj": var_obj,
                "vmec_failed": int(vmec_failed), "tracing_failed": int(tracing_failed),
                "eval_time_s": eval_time,
            }, points_fieldnames)

        # --- Reseed comparison: only meaningful for case-1 points, where
        # var_obj is a hardcoded constant rather than a measured quantity. ---
        if case == 1 and not vmec_failed and not tracing_failed and p_idx not in completed_reseed:
            rng = np.random.default_rng([args.random_seed, 900_000 + p_idx])
            seed_a, seed_b = 0, 0
            while seed_a == seed_b or seed_a == 8 or seed_b == 8:
                seed_a, seed_b = (int(v) for v in rng.integers(1, 1_000_000, size=2))
            print(f"[point {p_idx}] case 1 -> reseed comparison with seed_a={seed_a}, seed_b={seed_b}")
            loss_a, case_a, var_obj_a, tracing_failed_a = evaluate_reseed(cfg, eval_dir_base, seed_a)
            loss_b, case_b, var_obj_b, tracing_failed_b = evaluate_reseed(cfg, eval_dir_base, seed_b)
            diff = (loss_a - loss_b) if loss_a is not None and loss_b is not None else None
            print(f"[point {p_idx}] reseed: loss_a={loss_a} (case={case_a})  "
                  f"loss_b={loss_b} (case={case_b})  diff={diff}")
            append_result_row(reseed_csv, {
                "point": p_idx, "seed_a": seed_a, "seed_b": seed_b,
                "loss_a": loss_a, "loss_b": loss_b, "diff": diff,
                "case_a": case_a, "case_b": case_b,
                "tracing_failed_a": int(tracing_failed_a), "tracing_failed_b": int(tracing_failed_b),
            }, reseed_fieldnames)

        # --- Directional central-difference diagnostic, n_random_directions per point ---
        for k in range(args.n_random_directions):
            if (p_idx, k) in completed_directions:
                continue
            global_dir_idx = p_idx * 10_000 + k  # keep z distinct across points and directions
            h, z, clipped = sample_direction_step(
                global_dir_idx, args.random_seed, dim, lengthscale_p, args.perturb_scale, x_p, x_min
            )
            h_norm = float(np.linalg.norm(h))
            eval_dir_plus = os.path.join(eval_root, f"pt{p_idx:02d}_{run.name}_dir{k:02d}_plus")
            eval_dir_minus = os.path.join(eval_root, f"pt{p_idx:02d}_{run.name}_dir{k:02d}_minus")

            t0 = time.time()
            loss_plus, case_p, var_obj_p, vmec_failed_p, tracing_failed_p = evaluate_direction(
                objective, x_p + h, eval_dir_plus
            )
            loss_minus, case_m, var_obj_m, vmec_failed_m, tracing_failed_m = evaluate_direction(
                objective, x_p - h, eval_dir_minus
            )
            eval_time = time.time() - t0

            vmec_failed_dir = vmec_failed_p or vmec_failed_m
            tracing_failed_dir = tracing_failed_p or tracing_failed_m

            if loss_plus is not None and loss_minus is not None and args.perturb_scale > 0:
                gradient = (loss_plus - loss_minus) / (2.0 * args.perturb_scale)  # z^T D grad
            else:
                gradient = None

            append_result_row(directional_csv, {
                "point": p_idx, "direction": k, "step_norm": h_norm,
                "loss_plus": loss_plus, "loss_minus": loss_minus, "gradient": gradient,
                "case_p": case_p, "var_obj_p": var_obj_p, "case_m": case_m, "var_obj_m": var_obj_m,
                "clipped": int(clipped), "vmec_failed": int(vmec_failed_dir), "tracing_failed": int(tracing_failed_dir),
                "eval_time_s": eval_time, "h_json": json.dumps(h.tolist()), "z": json.dumps(z.tolist()),
            }, directional_fieldnames)
            print(f"[point {p_idx}][dir {k}] step_norm={h_norm:.4e}  loss_plus={loss_plus}  "
                  f"loss_minus={loss_minus}  z^T D grad={gradient}  "
                  f"({eval_time:.1f}s, clipped={clipped})")

    print(f"\nDone. Wrote {points_csv}, {reseed_csv}, {directional_csv}")


if __name__ == "__main__":
    main()
