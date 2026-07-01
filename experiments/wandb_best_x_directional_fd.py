#!/usr/bin/env python
"""Real-objective directional finite differences around the best x found
across a wandb project.

For the run/iteration with the global minimum logged `loss` in a wandb
project of experiments/botorch_optimize.py runs, samples N random directions
in DOF space -- weighted by the GP ARD lengthscale wandb logged at that
iteration -- and evaluates the TRUE (VMEC + alpha-particle-tracing) objective
at `x_best + h` for each direction (forward difference; `f(x_best)` is
already known exactly from the original wandb run, since it's a real
evaluation and not a surrogate estimate, so it isn't re-simulated).

This runs sequentially and can take a long time (VMEC + tracing per
evaluation, times --n-random-directions). Results are checkpointed to
`<output-dir>/directional_fd.csv` after every evaluation, so an interrupted
run can be resumed by rerunning with the same --output-dir and --random-seed.

Usage:
    python wandb_best_x_directional_fd.py --project garabedian_linear_cei_pressure
"""

import argparse
import csv
import json
import os
import re
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import yaml
from vmecpp.simsopt_compat import Vmec

from alpha_opt.objective import VmecConvergenceError
from alpha_opt.surface import SurfaceGarabedianLinear, SurfaceGarabedianQuantiles
from botorch_optimize import build_objective

DEFAULT_ENTITY = "sp2582-cornell-university"


# ---------------------------------------------------------------------------
# Step 1: find the global best loss across every run in the project
# ---------------------------------------------------------------------------

def fetch_run_dataframe(run):
    """Full (unsampled) logged history for a run as a DataFrame, or None."""
    rows = list(run.scan_history())
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "iteration" not in df.columns or "loss" not in df.columns:
        return None
    df = df.dropna(subset=["iteration", "loss"]).copy()
    if df.empty:
        return None
    df["iteration"] = df["iteration"].astype(int)
    for flag in ("vmec_failed", "tracing_failed"):
        if flag not in df.columns:
            df[flag] = 0.0
        df[flag] = df[flag].fillna(0.0)
    return df.sort_values("iteration").reset_index(drop=True)


def find_global_best(api, entity, project):
    path = f"{entity}/{project}" if entity else project
    runs = list(api.runs(path))
    print(f"Found {len(runs)} run(s) in '{path}'")

    best_run, best_row, best_df = None, None, None
    for run in runs:
        df = fetch_run_dataframe(run)
        if df is None:
            continue
        valid = df[(df["vmec_failed"] == 0) & (df["tracing_failed"] == 0) & np.isfinite(df["loss"])]
        if valid.empty:
            continue
        row = valid.loc[valid["loss"].idxmin()]
        if best_row is None or row["loss"] < best_row["loss"]:
            best_run, best_row, best_df = run, row, df

    if best_run is None:
        raise RuntimeError(f"No run in '{path}' has any valid logged 'loss'.")

    i_star = int(best_row["iteration"])
    print(f"Global best: run='{best_run.name}', iteration={i_star}, loss={best_row['loss']:.6e}")
    return best_run, best_row, best_df, i_star


# ---------------------------------------------------------------------------
# Step 2: reconstruct x_best and lengthscale via checkpoint artifacts
# ---------------------------------------------------------------------------

def load_state_from_checkpoint(run, min_len):
    """Return (state_dict, checkpoint_n) for the smallest checkpoint whose
    dense `results` list covers at least `min_len` trials, or the largest
    available checkpoint as a best-effort fallback (with a warning) if none
    is big enough."""
    candidates = []
    for art in run.logged_artifacts():
        if art.type != "checkpoint":
            continue
        m = re.match(r"state_(\d+)", art.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), art))
    candidates.sort(key=lambda t: t[0])
    if not candidates:
        return None, None

    for n, art in candidates:
        if n >= min_len:
            with tempfile.TemporaryDirectory() as d:
                local_dir = art.download(root=d)
                with open(os.path.join(local_dir, "botorch_state.json")) as f:
                    state = json.load(f)
            if len(state["results"]) >= min_len:
                return state, n

    n, art = candidates[-1]
    print(f"WARNING: no checkpoint covers trial {min_len - 1}; falling back to the "
          f"largest available checkpoint (state_{n:06d}).")
    with tempfile.TemporaryDirectory() as d:
        local_dir = art.download(root=d)
        with open(os.path.join(local_dir, "botorch_state.json")) as f:
            state = json.load(f)
    return state, n


def position_in_checkpoint(results, iteration, use_gp_constraints):
    """Map a dense wandb/trial 'iteration' index to its (possibly compacted)
    position in the checkpoint's X/Y arrays.

    botorch_optimize.py logs one wandb row per trial (success or failure),
    but when cfg["use_gp_constraints"] is True, only *successful* trials are
    ever appended to X_t/Y_t (failures go only to the constraint arrays
    Xc_t/C_t). state["results"] is dense -- one entry per attempted trial, in
    order -- so counting successes up to `iteration` recovers the correct
    compacted position.
    """
    trials = results[: iteration + 1]
    if trials[-1]["vmec_failed"] or trials[-1]["tracing_failed"]:
        raise ValueError(f"Trial {iteration} is recorded as failed in the checkpoint; expected a success.")
    if not use_gp_constraints:
        return iteration
    n_success = sum(1 for r in trials if not r["vmec_failed"] and not r["tracing_failed"])
    return n_success - 1


def get_x_best_and_lengthscale(best_run, best_row, i_star, use_gp_constraints):
    state, ckpt_n = load_state_from_checkpoint(best_run, i_star + 1)
    if state is None:
        raise RuntimeError(f"Run '{best_run.name}' has no 'checkpoint' artifacts; cannot recover x_best.")

    param_names = state["param_names"]
    X_full = np.asarray(state["X"], dtype=float)
    Y_full = np.asarray(state["Y"], dtype=float)
    print(f"Loaded checkpoint 'state_{ckpt_n:06d}': {len(X_full)} X/Y entries, "
          f"{len(state['results'])} trials total, dim={X_full.shape[1]}")

    pos = position_in_checkpoint(state["results"], i_star, use_gp_constraints)
    x_best = X_full[pos]
    y_best = float(Y_full[pos])
    print(f"Trial iteration {i_star} -> compacted position {pos} in checkpoint X/Y")
    print(f"Cross-check: checkpoint loss={y_best:.6e}  |  wandb-logged loss={best_row['loss']:.6e}  |  "
          f"results[{i_star}]['loss']={state['results'][i_star]['loss']:.6e}")

    ls_cols = sorted(
        (c for c in best_row.index if c.startswith("gp/lengthscale_")),
        key=lambda c: int(c.split("_")[-1]),
    )
    lengthscale_logged = best_row[ls_cols].to_numpy(dtype=float)
    if len(lengthscale_logged) != len(x_best):
        raise RuntimeError(
            f"lengthscale dim ({len(lengthscale_logged)}) != x_best dim ({len(x_best)}); "
            "wandb run and checkpoint disagree on dimensionality."
        )
    return x_best, param_names, lengthscale_logged, y_best


# ---------------------------------------------------------------------------
# Step 3: build the real objective (VMEC + tracing), matching
# botorch_optimize.py's main()
# ---------------------------------------------------------------------------

def build_real_objective(cfg):
    aspect_ratio = cfg["aspect_ratio"]
    minor_radius = (
        float(cfg["minor_radius"])
        if cfg.get("minor_radius") is not None
        else 3.1 / aspect_ratio ** 0.38
    )
    major_radius = minor_radius * aspect_ratio

    vmec = Vmec(cfg["vmec_input_file"], verbose=True)
    avg_B_estimate = cfg["max_B_target"] / np.sqrt(2)
    phiedge_high = np.pi * avg_B_estimate * minor_radius ** 2 * 2
    vmec.set("phiedge", phiedge_high)

    if cfg["parameterization"] == "garabedian_quantiles":
        surface = SurfaceGarabedianQuantiles(
            vmec.indata.nfp, mpol=cfg["mpol"], ntor=cfg["ntor"],
            minor_radius=minor_radius, major_radius=major_radius,
            filename=cfg["data_file"], exact_radii=True,
        )
    elif cfg["parameterization"] == "garabedian_linear":
        surface = SurfaceGarabedianLinear(
            vmec.indata.nfp, mpol=cfg["mpol"], ntor=cfg["ntor"],
            minor_radius=minor_radius, major_radius=major_radius,
            filename=cfg["data_file"], exact_radii=True,
        )
    else:
        raise ValueError(f"Unsupported reparameterization: {cfg['parameterization']}")

    objective = build_objective(cfg, vmec, surface, phiedge_high)
    return objective, len(surface.x)


# ---------------------------------------------------------------------------
# Step 4: random directions + forward-difference evaluation
# ---------------------------------------------------------------------------

def sample_direction_step(direction_idx, random_seed, dim, lengthscale, perturb_scale, x_best, x_min):
    """h_j = perturb_scale * lengthscale_j * z_j, clipped per-dimension so
    x_best + h stays within [x_min, 1 - x_min] (only the + side needs to stay
    in bounds for a forward difference)."""
    rng = np.random.default_rng([random_seed, direction_idx])
    z = rng.standard_normal(dim)
    h = perturb_scale * lengthscale * z
    room_hi = (1.0 - x_min) - x_best
    room_lo = x_best - x_min
    return np.clip(h, -room_lo, room_hi)


def evaluate_direction(objective, x_trial, eval_dir):
    os.makedirs(eval_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(eval_dir)
    try:
        loss, _case, _var_obj = objective(x_trial)
        return float(loss), False, False
    except VmecConvergenceError:
        return None, True, False
    except Exception:
        return None, False, True
    finally:
        os.chdir(cwd)


def append_result_row(csv_path, row, fieldnames):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_completed_directions(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        return {int(row["direction"]) for row in csv.DictReader(f)}


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
                         help="fraction of each dimension's lengthscale used as the FD step size")
    parser.add_argument("--n-random-directions", type=int, default=200,
                         help="number of random lengthscale-weighted directions to evaluate")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="directional_fd_results",
                         help="where to write the checkpointed CSV, summary JSON, and plot")
    parser.add_argument("--no-validate-x-best", action="store_true",
                         help="skip the extra real evaluation at x_best that cross-checks the "
                              "reconstructed VMEC/surface setup against the wandb-logged loss")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "directional_fd.csv")
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

    # Reference loss scale for "is this a sharp spike" -- std of the run's own
    # valid logged loss up to the best iteration (no GP fit needed).
    valid_hist = best_df[
        (best_df["vmec_failed"] == 0) & (best_df["tracing_failed"] == 0) & (best_df["iteration"] < i_star)
    ]
    loss_std = float(valid_hist["loss"].std()) if len(valid_hist) > 1 else float("nan")
    print(f"Loss std over run history up to iteration {i_star}: {loss_std:.6e}")

    objective, dim = build_real_objective(cfg)
    if dim != len(x_best):
        raise RuntimeError(
            f"Surface built from --config-path has dim={dim}, but x_best from wandb/checkpoint "
            f"has dim={len(x_best)}. --config-path likely doesn't match this project's runs."
        )
    x_min = float(cfg["x_min"])

    if not args.no_validate_x_best:
        print("\nValidating reconstructed setup: evaluating the real objective at x_best...")
        t0 = time.time()
        loss_check, vmec_failed, tracing_failed = evaluate_direction(
            objective, x_best, os.path.join(eval_root, f"{best_run.name}_xbest_check")
        )
        print(f"objective(x_best) = {loss_check}  (wandb-logged: {y_best:.6e})  "
              f"[{time.time() - t0:.1f}s, vmec_failed={vmec_failed}, tracing_failed={tracing_failed}]")
        if loss_check is not None and not np.isclose(loss_check, y_best, rtol=1e-3, atol=1e-6):
            print("WARNING: objective(x_best) does not match the wandb-logged loss within tolerance -- "
                  "the reconstructed VMEC/surface/config setup may not exactly match the original run.")

    fieldnames = ["direction", "step_norm", "loss_plus", "gradient", "delta_over_std",
                  "vmec_failed", "tracing_failed", "eval_time_s", "h_json"]
    done = load_completed_directions(csv_path)
    if done:
        print(f"\nResuming: {len(done)} direction(s) already evaluated in {csv_path}")

    print(f"\nEvaluating {args.n_random_directions} random directions "
          f"({args.n_random_directions - len(done)} remaining)...")
    for k in range(args.n_random_directions):
        if k in done:
            continue
        h = sample_direction_step(k, args.random_seed, dim, lengthscale_logged, args.perturb_scale, x_best, x_min)
        h_norm = float(np.linalg.norm(h))
        eval_dir = os.path.join(eval_root, f"{best_run.name}_dir{k:04d}")

        t0 = time.time()
        loss_plus, vmec_failed, tracing_failed = evaluate_direction(objective, x_best + h, eval_dir)
        eval_time = time.time() - t0

        if loss_plus is not None and h_norm > 0:
            gradient = (loss_plus - y_best) / h_norm
            delta_over_std = (loss_plus - y_best) / loss_std if loss_std else float("nan")
        else:
            gradient, delta_over_std = None, None

        row = {
            "direction": k, "step_norm": h_norm, "loss_plus": loss_plus, "gradient": gradient,
            "delta_over_std": delta_over_std, "vmec_failed": vmec_failed, "tracing_failed": tracing_failed,
            "eval_time_s": eval_time, "h_json": json.dumps(h.tolist()),
        }
        append_result_row(csv_path, row, fieldnames)
        print(f"[{k + 1}/{args.n_random_directions}] step_norm={h_norm:.4e}  loss_plus={loss_plus}  "
              f"gradient={gradient}  ({eval_time:.1f}s, vmec_failed={vmec_failed}, tracing_failed={tracing_failed})")

    # --- Summary ---
    results_df = pd.read_csv(csv_path)
    valid = results_df[(results_df["vmec_failed"] == False) & (results_df["tracing_failed"] == False)
                        & results_df["loss_plus"].notna()].copy()
    print(f"\n{len(valid)}/{len(results_df)} directions evaluated successfully.")

    sharp = valid[valid["delta_over_std"].abs() >= 1.0]
    frac_sharp = len(sharp) / len(valid) if len(valid) else float("nan")
    print(f"Run '{best_run.name}', iteration {i_star}, best loss (wandb) = {y_best:.6e}")
    print(f"{len(sharp)}/{len(valid)} directions ({frac_sharp:.0%}) show |delta_over_std| >= 1 "
          "(real-objective evaluation, forward difference).")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "run": best_run.name,
            "iteration": i_star,
            "x_best": x_best.tolist(),
            "param_names": param_names,
            "y_best": y_best,
            "lengthscale_logged": lengthscale_logged.tolist(),
            "loss_std_reference": loss_std,
            "n_directions_requested": args.n_random_directions,
            "n_evaluated_successfully": int(len(valid)),
            "n_sharp": int(len(sharp)),
            "frac_sharp": float(frac_sharp) if len(valid) else None,
        }, f, indent=2)
    print(f"Wrote {summary_path}")

    if len(valid):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].hist(valid["gradient"].abs(), bins=25, color="C0")
        axes[0].set_xlabel("|directional gradient|  (loss / unit x, real objective)")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"Distribution over {len(valid)} real-objective directional evals")

        axes[1].hist(valid["delta_over_std"], bins=25, color="C1")
        axes[1].axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("delta_over_std  ((loss_plus - y_best) / loss_std)")
        axes[1].set_ylabel("count")
        axes[1].set_title("Sharpness distribution")

        fig.tight_layout()
        plot_path = os.path.join(args.output_dir, "directional_fd_summary.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
