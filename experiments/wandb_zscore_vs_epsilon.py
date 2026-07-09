#!/usr/bin/env python
"""Sweep the case-2 alpha-loss objective's `epsilon` and see how it changes
the GP surrogate's sequential (leave-future-out) Z-score calibration for a
single wandb run -- without re-running any VMEC/tracing evaluations.

Why this is possible without re-simulating: alpha_loss_objective_from_times's
case-2 branch (alpha_opt/tracing/loss_times.py) is

    loss = log10(ELF + eps) - log10(t_max) - log10(threshold + eps)

which is invertible in ELF given the logged loss, t_max, threshold, and the
epsilon that was actually used when the run happened (the default,
1/n_particles). So for every case-2 trial we recover ELF exactly, then
recompute what `loss` (and its delta-method variance `var_obj`) would have
been under a different epsilon. Case-1 trials (`loss = -log10(t_exceed)`)
don't depend on epsilon at all and pass through unchanged. Which case a
trial belongs to is inferred purely from the loss value: case-2 objectives
are always <= -log10(t_max), case-1 objectives are always > -log10(t_max)
(this also lets it work for the initial Sobol batch, which never gets a
`case` field logged to wandb -- see the checkpoint-artifact fallback below).
This requires threshold (cfg["maxloss"]) < 1, i.e. case-0 never occurs; the
script warns if that's not true.

For each swept epsilon value, this replays the *exact* sequential
GP-fit-and-predict procedure botorch_optimize.py used online (matching
propose_next()'s unconstrained SingleTaskGP: same kernel, input/outcome
transforms, Normalize bounds): at each BO iteration i, fit on all prior
compacted training rows [0, pos(i)), predict at trial i's x, and compute
    z = (loss_new[i] - pred_mean) / pred_std
This exactly reproduces the already-logged `z_score` wandb field when
epsilon equals the original value (used as a sanity check, plotted as a
reference line), and shows how that calibration diagnostic would change
under a different epsilon.

NOTE: only the objective GP is replicated (matching the unconstrained
SingleTaskGP construction inside propose_next()); the separate constraint GP
used when cfg["use_gp_constraints"] is True doesn't affect the objective's
predictive mean/variance and isn't reproduced here. The initial Sobol design
phase is never logged as `z_score` by botorch_optimize.py (propose_next()
isn't called for it), so Z-scores are only reported starting at iteration
`num_initial`, matching what's actually comparable to the logged field.

Training data (X/Y/Y_var per compacted position, and the dense per-trial
vmec_failed/tracing_failed flags needed to replay the compaction) comes from
the run's largest `checkpoint` wandb artifact, not from scan_history -- the
initial Sobol batch is never logged as history rows at all.

This is slow: it fits one GP per (epsilon, iteration) pair. Progress is
checkpointed to `<output-dir>/zscore_vs_epsilon.csv` after every fit, so an
interrupted run can be resumed by rerunning with the same --output-dir. Use
--stride to skip iterations for a coarser/faster sweep.

Usage:
    python wandb_zscore_vs_epsilon.py --project garabedian_linear_cei_pressure \\
        --run solar-comet-12 --config-path ../configs/garabedian_linear_pressure.yaml
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
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.mlls import ExactMarginalLogLikelihood

DEFAULT_ENTITY = "sp2582-cornell-university"
LN10 = np.log(10.0)


# ---------------------------------------------------------------------------
# Step 1: pull the dense per-trial history from the run's checkpoint artifact
# (NOT scan_history -- the initial Sobol batch is never logged as rows).
# ---------------------------------------------------------------------------

def find_run(api, entity, project, run_name):
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"display_name": run_name})
    if not runs:
        raise ValueError(f"No wandb run found with name '{run_name}' in '{path}'")
    return runs[0]


def load_full_checkpoint(run):
    """Return the state dict from the largest available 'checkpoint' artifact."""
    candidates = []
    for art in run.logged_artifacts():
        if art.type != "checkpoint":
            continue
        m = re.match(r"state_(\d+)", art.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), art))
    if not candidates:
        raise RuntimeError(f"Run '{run.name}' has no 'checkpoint' artifacts.")
    candidates.sort(key=lambda t: t[0])
    n, art = candidates[-1]
    with tempfile.TemporaryDirectory() as d:
        local_dir = art.download(root=d)
        with open(os.path.join(local_dir, "botorch_state.json")) as f:
            state = json.load(f)
    print(f"Loaded checkpoint 'state_{n:06d}': {len(state['results'])} dense trials, "
          f"{len(state['X'])} compacted X/Y rows.")
    return state


def compacted_position(results, use_gp_constraints, dense_idx):
    """Position of dense trial `dense_idx` in the compacted X/Y/Y_var arrays,
    or None if it was excluded entirely (a failed trial when
    use_gp_constraints=True, which only ever goes to the constraint arrays)."""
    trials = results[: dense_idx + 1]
    is_failed = trials[-1]["vmec_failed"] or trials[-1]["tracing_failed"]
    if not use_gp_constraints:
        return dense_idx
    if is_failed:
        return None
    return sum(1 for r in trials if not (r["vmec_failed"] or r["tracing_failed"])) - 1


# ---------------------------------------------------------------------------
# Step 2: infer case-2 membership and recover ELF, then recompute loss/var
# for an arbitrary epsilon.
# ---------------------------------------------------------------------------

def infer_case2_and_recover_elf(Y, Yvar, threshold, t_max, eps_orig):
    """Case-2 objectives are always <= -log10(t_max); case-1 are always
    strictly greater (see module docstring). For case-2 rows, invert
    loss = log10(ELF+eps) - log10(t_max) - log10(threshold+eps) for ELF,
    and invert the delta-method variance for var(ELF)."""
    case_boundary = -np.log10(t_max)
    is_case2 = Y <= case_boundary + 1e-9

    ELF = np.full_like(Y, np.nan)
    var_ELF = np.full_like(Y, np.nan)

    elf2 = t_max * (threshold + eps_orig) * (10.0 ** Y[is_case2]) - eps_orig
    elf2 = np.maximum(elf2, 0.0)
    h_prime_old = 1.0 / ((elf2 + eps_orig) * LN10)

    ELF[is_case2] = elf2
    var_ELF[is_case2] = Yvar[is_case2] / (h_prime_old ** 2)
    return is_case2, ELF, var_ELF


def recompute_loss_var(Y, Yvar, is_case2, ELF, var_ELF, threshold, t_max, eps_new):
    Y_new = Y.copy()
    Yvar_new = Yvar.copy()
    elf2 = ELF[is_case2]
    Y_new[is_case2] = np.log10(elf2 + eps_new) - np.log10(t_max) - np.log10(threshold + eps_new)
    h_prime_new = 1.0 / ((elf2 + eps_new) * LN10)
    Yvar_new[is_case2] = h_prime_new ** 2 * var_ELF[is_case2]
    return Y_new, Yvar_new


# ---------------------------------------------------------------------------
# Step 3: sequential (leave-future-out) GP fit + Z-score, matching
# propose_next()'s unconstrained SingleTaskGP construction exactly.
# ---------------------------------------------------------------------------

def fit_and_predict(train_X, train_Y, train_Yvar, test_x, bounds, use_rbf):
    model = SingleTaskGP(
        train_X,
        (-train_Y).unsqueeze(-1),  # matches propose_next(): negated for max, un-negated below
        train_Yvar=train_Yvar.unsqueeze(-1),
        covar_module=get_covar_module_with_dim_scaled_prior(ard_num_dims=train_X.shape[-1], use_rbf_kernel=use_rbf),
        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    with torch.no_grad():
        posterior = model.posterior(test_x.unsqueeze(0))
        pred_mean = -posterior.mean.squeeze().item()
        pred_var = posterior.variance.squeeze().item()
    return pred_mean, pred_var


def append_result_row(csv_path, row, fieldnames):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_done_pairs(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        return {(float(row["epsilon"]), int(row["iteration"])) for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="wandb entity (team/user)")
    parser.add_argument("--run", required=True, help="wandb run (display) name")
    parser.add_argument("--config-path", required=True,
                         help="local yaml config matching this run (t_max, maxloss, x_min, kernel, "
                              "use_gp_constraints, num_initial, n_particles)")
    parser.add_argument("--epsilon-original", type=float, default=None,
                         help="epsilon actually used when the run happened (default: 1/cfg['n_particles'])")
    parser.add_argument("--epsilon-min", type=float, default=1e-7)
    parser.add_argument("--epsilon-max", type=float, default=1e-1)
    parser.add_argument("--n-epsilon", type=int, default=5, help="number of log-spaced epsilon values")
    parser.add_argument("--epsilons", type=str, default=None,
                         help="comma-separated explicit epsilon values; overrides --epsilon-min/-max/-n-epsilon")
    parser.add_argument("--min-train-points", type=int, default=None,
                         help="dense iteration to start reporting Z-scores at (default: cfg['num_initial'], "
                              "matching what botorch_optimize.py actually logged z_score for)")
    parser.add_argument("--stride", type=int, default=5, help="only compute every --stride-th iteration")
    parser.add_argument("--output-dir", default="epsilon_zscore_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "zscore_vs_epsilon.csv")

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)
    threshold = float(cfg["maxloss"])
    t_max = float(cfg["t_max"])
    x_min = float(cfg["x_min"])
    use_gp_constraints = bool(cfg["use_gp_constraints"])
    use_rbf = cfg["kernel"] == "rbf"
    min_train_points = args.min_train_points if args.min_train_points is not None else int(cfg["num_initial"])
    eps_orig = args.epsilon_original if args.epsilon_original is not None else 1.0 / float(cfg["n_particles"])

    if threshold >= 1.0:
        print(f"WARNING: cfg['maxloss']={threshold} >= 1.0, so this run's objective is always case 0 "
              "(no epsilon dependence at all) -- this sweep won't show anything.")

    api = wandb.Api()
    run = find_run(api, args.entity, args.project, args.run)
    state = load_full_checkpoint(run)

    param_names = state["param_names"]
    X_full = np.asarray(state["X"], dtype=float)
    Y_full = np.asarray(state["Y"], dtype=float)
    Yvar_full = np.asarray(state["Y_var"], dtype=float)
    results = state["results"]
    dim = X_full.shape[1]
    print(f"dim={dim}, param_names={param_names}")
    print(f"eps_original={eps_orig:.6e}  threshold(maxloss)={threshold}  t_max={t_max}  "
          f"use_gp_constraints={use_gp_constraints}  min_train_points={min_train_points}")

    is_case2, ELF, var_ELF = infer_case2_and_recover_elf(Y_full, Yvar_full, threshold, t_max, eps_orig)
    print(f"{is_case2.sum()}/{len(is_case2)} compacted rows inferred as case 2 "
          f"({is_case2.mean():.0%})")

    # Sanity check: reversing then recomputing at eps_original must reproduce
    # the original logged loss/var_obj almost exactly.
    Y_check, Yvar_check = recompute_loss_var(Y_full, Yvar_full, is_case2, ELF, var_ELF, threshold, t_max, eps_orig)
    max_loss_err = np.nanmax(np.abs(Y_check - Y_full))
    max_var_err = np.nanmax(np.abs(Yvar_check[is_case2] - Yvar_full[is_case2])) if is_case2.any() else 0.0
    print(f"Reversal sanity check (should be ~0): max|loss_reconstructed - loss_orig|={max_loss_err:.3e}, "
          f"max|var_reconstructed - var_orig|={max_var_err:.3e}")
    if max_loss_err > 1e-6:
        print("WARNING: reversal does not reproduce the original loss within tolerance -- "
              "check --epsilon-original and --config-path match this run.")

    bounds = torch.stack([
        torch.full((dim,), x_min, dtype=torch.double),
        torch.full((dim,), 1.0 - x_min, dtype=torch.double),
    ])

    if args.epsilons:
        epsilons = sorted({float(v) for v in args.epsilons.split(",")} | {eps_orig})
    else:
        epsilons = sorted(set(np.geomspace(args.epsilon_min, args.epsilon_max, args.n_epsilon).tolist()) | {eps_orig})
    print(f"Epsilon sweep ({len(epsilons)} values): {['%.3e' % e for e in epsilons]}")

    n_dense = len(results)
    iterations = list(range(min_train_points, n_dense, args.stride))
    print(f"\nSweeping {len(epsilons)} epsilon(s) x {len(iterations)} iteration(s) "
          f"(dense trials {min_train_points}..{n_dense - 1}, stride={args.stride})...")

    fieldnames = ["epsilon", "iteration", "pos", "is_case2", "loss_new", "pred_mean", "pred_std", "z_score"]
    done = load_done_pairs(csv_path)
    if done:
        print(f"Resuming: {len(done)} (epsilon, iteration) pair(s) already done in {csv_path}")

    t_start = time.time()
    n_total = len(epsilons) * len(iterations)
    n_done_now = 0
    for eps in epsilons:
        Y_new, Yvar_new = recompute_loss_var(Y_full, Yvar_full, is_case2, ELF, var_ELF, threshold, t_max, eps)
        for i in iterations:
            if (eps, i) in done:
                continue
            pos = compacted_position(results, use_gp_constraints, i)
            if pos is None or pos < 2:
                continue

            train_X = torch.tensor(X_full[:pos], dtype=torch.double)
            train_Y = torch.tensor(Y_new[:pos], dtype=torch.double)
            train_Yvar = torch.tensor(Yvar_new[:pos], dtype=torch.double)
            test_x = torch.tensor(X_full[pos], dtype=torch.double)

            try:
                pred_mean, pred_var = fit_and_predict(train_X, train_Y, train_Yvar, test_x, bounds, use_rbf)
                pred_std = pred_var ** 0.5 if pred_var > 0 else float("nan")
                z_score = (Y_new[pos] - pred_mean) / pred_std if np.isfinite(pred_std) and pred_std > 0 else float("nan")
            except Exception as e:
                print(f"  eps={eps:.3e} iter={i}: GP fit failed ({e}); skipping")
                pred_mean, pred_std, z_score = float("nan"), float("nan"), float("nan")

            row = {
                "epsilon": eps, "iteration": i, "pos": pos, "is_case2": bool(is_case2[pos]),
                "loss_new": float(Y_new[pos]), "pred_mean": pred_mean, "pred_std": pred_std, "z_score": z_score,
            }
            append_result_row(csv_path, row, fieldnames)
            n_done_now += 1
            if n_done_now % 20 == 0:
                elapsed = time.time() - t_start
                rate = n_done_now / elapsed
                remaining = (n_total - len(done) - n_done_now) / rate if rate > 0 else float("nan")
                print(f"  [{len(done) + n_done_now}/{n_total}] eps={eps:.3e} iter={i} z={z_score:.3f} "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # --- Plots + summary ---
    df = pd.read_csv(csv_path)
    df = df[np.isfinite(df["z_score"])]

    fig, ax = plt.subplots(figsize=(9, 6))
    eps_values = sorted(df["epsilon"].unique())
    colors = cm.viridis(np.linspace(0, 1, len(eps_values)))
    for eps, color in zip(eps_values, colors):
        sub = df[df["epsilon"] == eps].sort_values("iteration")
        label = f"eps={eps:.2e}" + ("  (original)" if np.isclose(eps, eps_orig) else "")
        lw = 2.2 if np.isclose(eps, eps_orig) else 1.0
        ax.plot(sub["iteration"], sub["z_score"], marker="o", markersize=3, linewidth=lw, color=color, label=label)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(2.0, color="red", linewidth=0.6, linestyle=":")
    ax.axhline(-2.0, color="red", linewidth=0.6, linestyle=":")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Z-score")
    ax.set_title(f"Sequential Z-score vs iteration, by epsilon\nrun='{args.run}'")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    plot1_path = os.path.join(args.output_dir, "zscore_vs_iteration.png")
    fig.savefig(plot1_path, dpi=150)
    print(f"\nWrote {plot1_path}")

    summary = df.groupby("epsilon")["z_score"].agg(
        rms_abs_z=lambda s: float(np.sqrt(np.mean(s ** 2))),
        mean_z="mean",
        frac_abs_gt_2=lambda s: float((s.abs() >= 2.0).mean()),
        n="count",
    ).reset_index().sort_values("epsilon")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    axes2[0].plot(summary["epsilon"], summary["rms_abs_z"], marker="o")
    axes2[0].axvline(eps_orig, color="gray", linewidth=0.8, linestyle="--", label="original epsilon")
    axes2[0].set_xscale("log")
    axes2[0].set_xlabel("epsilon")
    axes2[0].set_ylabel("RMS |Z-score|")
    axes2[0].set_title("Overall calibration vs epsilon\n(1.0 = well-calibrated)")
    axes2[0].axhline(1.0, color="green", linewidth=0.8, linestyle=":")
    axes2[0].legend(fontsize=8)

    axes2[1].plot(summary["epsilon"], summary["frac_abs_gt_2"], marker="o", color="C1")
    axes2[1].axvline(eps_orig, color="gray", linewidth=0.8, linestyle="--", label="original epsilon")
    axes2[1].set_xscale("log")
    axes2[1].set_xlabel("epsilon")
    axes2[1].set_ylabel("fraction |Z| >= 2")
    axes2[1].set_title("Fraction of surprising observations vs epsilon")
    axes2[1].legend(fontsize=8)

    fig2.tight_layout()
    plot2_path = os.path.join(args.output_dir, "zscore_summary_vs_epsilon.png")
    fig2.savefig(plot2_path, dpi=150)
    print(f"Wrote {plot2_path}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "run": args.run, "epsilon_original": eps_orig, "epsilons": eps_values,
            "threshold": threshold, "t_max": t_max, "min_train_points": min_train_points,
            "summary_by_epsilon": summary.to_dict(orient="records"),
        }, f, indent=2)
    print(f"Wrote {summary_path}")
    print(f"\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
