#!/usr/bin/env python
"""Gather every case-1 data point logged across all runs (random seeds) in a
wandb project of experiments/botorch_optimize.py runs, and save it to a
single .npz file.

"case 1" is the alpha_loss_objective_from_times branch where the energy-loss
threshold is exceeded before t_max (see
alpha_opt/tracing/loss_times.py:alpha_loss_objective_from_times); its
`var_obj` is a hardcoded placeholder (1e-5), not a real delta-method
variance, unlike case 2. Each run in the project corresponds to one random
seed (botorch_optimize.py sets cfg["seed"] = wandb_run.config["seed"]), so
pooling case==1 rows across every run in the project pools across seeds.

Usage:
    python wandb_gather_case1.py --project garabedian_linear_cei_pressure \
        --output case1_data.npz
"""

import argparse

import numpy as np
import pandas as pd
import wandb

DEFAULT_ENTITY = "sp2582-cornell-university"


def fetch_run_dataframe(run, case_key):
    """Full (unsampled) logged history for a run as a DataFrame, or None."""
    rows = list(run.scan_history())
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "iteration" not in df.columns or "loss" not in df.columns or case_key not in df.columns:
        return None
    df = df.dropna(subset=["iteration", "loss", case_key]).copy()
    if df.empty:
        return None
    df["iteration"] = df["iteration"].astype(int)
    return df.sort_values("iteration").reset_index(drop=True)


def gather_case1(api, entity, project, param_prefix, case_key, metric, var_obj_key, worst_dmerc_key):
    """Pool every (x, loss, ...) row with case == 1 from every run in the
    project, tagging each row with the run's name and seed."""
    path = f"{entity}/{project}" if entity else project
    runs = list(api.runs(path))
    print(f"Found {len(runs)} run(s) in '{path}'")

    param_names = None
    X_list, loss_list, var_obj_list, worst_dmerc_list = [], [], [], []
    iteration_list, seed_list, run_name_list = [], [], []

    for run in runs:
        df = fetch_run_dataframe(run, case_key)
        if df is None:
            continue
        case1 = df[df[case_key].round().astype(int) == 1]
        if case1.empty:
            continue

        keys = sorted(k for k in case1.columns if k.startswith(param_prefix))
        if not keys:
            print(f"  WARNING: run '{run.name}' has case-1 rows but no '{param_prefix}*' columns; skipping.")
            continue
        if param_names is None:
            param_names = [k[len(param_prefix):] for k in keys]
        elif [k[len(param_prefix):] for k in keys] != param_names:
            print(f"  WARNING: run '{run.name}' has different parameter names; skipping.")
            continue

        seed = run.config.get("seed")
        n = len(case1)
        print(f"  run='{run.name}' (seed={seed}): {n} case-1 point(s)")

        X_list.append(case1[keys].to_numpy(dtype=float))
        loss_list.append(case1[metric].to_numpy(dtype=float))
        var_obj_list.append(
            case1[var_obj_key].to_numpy(dtype=float) if var_obj_key in case1.columns
            else np.full(n, np.nan)
        )
        worst_dmerc_list.append(
            case1[worst_dmerc_key].to_numpy(dtype=float) if worst_dmerc_key in case1.columns
            else np.full(n, np.nan)
        )
        iteration_list.append(case1["iteration"].to_numpy(dtype=int))
        seed_list.append(np.full(n, seed if seed is not None else -1, dtype=int))
        run_name_list.append(np.full(n, run.name, dtype=object))

    if not X_list:
        raise RuntimeError(f"No case-1 data points found in '{path}'.")

    return {
        "X": np.concatenate(X_list, axis=0),
        "loss": np.concatenate(loss_list, axis=0),
        "var_obj": np.concatenate(var_obj_list, axis=0),
        "worst_dmerc": np.concatenate(worst_dmerc_list, axis=0),
        "iteration": np.concatenate(iteration_list, axis=0),
        "seed": np.concatenate(seed_list, axis=0),
        "run_name": np.concatenate(run_name_list, axis=0),
        "param_names": np.array(param_names, dtype=object),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="wandb entity (team/user)")
    parser.add_argument("--metric", default="loss", help="objective metric key logged to wandb")
    parser.add_argument("--case-key", default="case", help="case-label key logged to wandb")
    parser.add_argument("--param-prefix", default="x/",
                         help="prefix of the design-parameter keys logged to wandb (e.g. 'x/name')")
    parser.add_argument("--var-obj-key", default="var_obj", help="observation-noise-variance key logged to wandb")
    parser.add_argument("--worst-dmerc-key", default="worst_dmerc", help="worst-DMerc key logged to wandb")
    parser.add_argument("--output", default="case1_data.npz", help="output .npz path")
    args = parser.parse_args()

    api = wandb.Api()
    data = gather_case1(
        api, args.entity, args.project, args.param_prefix, args.case_key,
        args.metric, args.var_obj_key, args.worst_dmerc_key,
    )

    print(f"\nPooled {len(data['X'])} case-1 point(s) across "
          f"{len(set(data['run_name'].tolist()))} run(s), dim={data['X'].shape[1]}")
    np.savez(args.output, **data)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
