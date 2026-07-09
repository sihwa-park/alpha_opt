#!/usr/bin/env python
"""Fit a low-rank subspace of the GP kernel's ARD metric to look for
low-dimensional structure in an already-explored wandb project.

The GP used throughout this project has a stationary ARD kernel
    k(x, y) = phi(||D(x - y)||_2),   D = diag(1 / lengthscale)
(Matern-5/2 by default, see `botorch_optimize.py`'s `propose_next`). This
script asks: is there a *low-rank* subspace U (a tall-thin d x k matrix with
orthonormal columns, U^T U = I_k) such that

    k(x, y) ~= phi(||U^T D (x - y)||_2)

fits the observed data about as well as the full-rank kernel? If so, the
objective effectively only varies along a k-dimensional subspace of the
d-dimensional DOF space.

Three parts, run in order by main():

  1. gather_data(): pool every (x, loss) pair from every run's fullest
     checkpoint in the wandb project, explicitly excluding any trial with
     vmec_failed or tracing_failed.

  2. fit_full_rank_gp() + fit_low_rank_kernel_newton():
       (a) refit a fresh full-rank ARD GP on the pooled data (same
           covar_module as botorch_optimize.py) to get D = diag(1/lengthscale).
       (b) with D fixed, solve for U (and the kernel's outputscale/noise)
           by maximizing the GP marginal log-likelihood, using a
           Newton-KKT (SQP) method: at each step, solve the linearized
           KKT system for the equality constraint c(U) = vec_utri(U^T U - I_k):

               [ H   J^T ] [dtheta]   [-g]
               [ J   0   ] [lambda] = [-c]

           where H, g are the (Levenberg-Marquardt-damped) Hessian/gradient
           of the negative log marginal likelihood w.r.t. theta = [vec(U);
           log_outputscale; log_noise], and J = dc/dtheta. This bakes the
           orthonormality constraint directly into the Newton system (rather
           than a first-order retraction after an unconstrained step); a
           polar-decomposition re-projection after each accepted step is
           kept only as a numerical safeguard against floating-point drift.
           Backtracking + damping provide a (very) approximate globalization
           since the KKT stationary point need not be a local minimum.

  3. Save U, D, the fitted hyperparameters, and the Newton iteration trace
     to --output-dir; print a summary comparing the full-rank and low-rank
     fits' negative log-likelihoods; plot leave-one-out (closed-form,
     Rasmussen & Williams eq. 5.12-5.13, no retraining) cross-validated
     predicted-vs-actual loss for both kernels side by side (the quantitative
     check of whether the rank-k subspace represents the data about as well
     as the full-rank kernel); and plot the data projected onto the fitted
     subspace (3D view + pairwise 2D projections for rank>=3), colored by
     loss, as a direct visual check of whether points cluster/organize by
     loss in that low-rank space.

Usage:
    python wandb_low_rank_kernel_fit.py --project garabedian_linear_cei_pressure --rank 3
"""

import argparse
import csv
import json
import os
import re
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.double)

DEFAULT_ENTITY = "sp2582-cornell-university"


# ---------------------------------------------------------------------------
# Part 1: gather all valid data points from the project
#
# fetch_run_dataframe/load_state_from_checkpoint are copied (not imported)
# from wandb_best_x_directional_cd.py: importing that module transitively
# pulls in botorch_optimize -> alpha_opt.tracing -> firm3dpp, which requires
# a GPU to even import. This script only does wandb I/O + a local GP fit, so
# it should run anywhere -- keeping these two helpers self-contained avoids
# an unrelated hard GPU dependency.
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

def gather_data(api, entity, project, use_gp_constraints, max_points, seed):
    """Pool every (x, loss) pair from every run's fullest checkpoint,
    excluding any trial with vmec_failed or tracing_failed.

    `state["results"]` is dense (one entry per attempted trial, with
    vmec_failed/tracing_failed flags). `state["X"]`/`state["Y"]` are only
    dense the same way when `use_gp_constraints` is False -- in that case
    botorch_optimize.py's main loop still appends FAILED trials to X_t/Y_t
    with a penalty loss (see the `else` branch around its `if not
    result.failed` check), so they must be filtered out here explicitly. When
    `use_gp_constraints` is True, failures are routed to Xc_t/C_t instead and
    X_t/Y_t are already success-only (compacted, shorter than `results`), so
    the success mask over `results` is applied positionally to match."""
    path = f"{entity}/{project}" if entity else project
    runs = list(api.runs(path))
    print(f"Found {len(runs)} run(s) in '{path}'")

    X_list, Y_list = [], []
    param_names = None
    for run in runs:
        df = fetch_run_dataframe(run)
        if df is None:
            continue
        state, ckpt_n = load_state_from_checkpoint(run, min_len=len(df))
        if state is None:
            continue
        X_full = np.asarray(state["X"], dtype=float)
        Y_full = np.asarray(state["Y"], dtype=float)
        if X_full.size == 0:
            continue

        success_mask = np.array(
            [not (r["vmec_failed"] or r["tracing_failed"]) for r in state["results"]]
        )
        n_success = int(success_mask.sum())
        if use_gp_constraints:
            # X_full/Y_full should already be success-only and compacted.
            if len(X_full) != n_success:
                print(f"  WARNING: run '{run.name}' expected {n_success} success-only "
                      f"X/Y entries (use_gp_constraints=True) but checkpoint has "
                      f"{len(X_full)}; skipping run to avoid mixing in failures.")
                continue
        else:
            # X_full/Y_full is dense (1 per trial); drop the failed rows,
            # which hold a penalty value in place of a real loss.
            if len(X_full) != len(success_mask):
                print(f"  WARNING: run '{run.name}' X/Y length ({len(X_full)}) != "
                      f"results length ({len(success_mask)}); skipping run.")
                continue
            X_full, Y_full = X_full[success_mask], Y_full[success_mask]

        if X_full.size == 0:
            continue
        if param_names is None:
            param_names = list(state["param_names"])
        elif list(state["param_names"]) != param_names:
            print(f"  WARNING: run '{run.name}' has different param_names; skipping.")
            continue
        print(f"  run='{run.name}': {len(X_full)} successful point(s) "
              f"(of {len(success_mask)} attempted) from checkpoint state_{ckpt_n:06d}")
        X_list.append(X_full)
        Y_list.append(Y_full)

    if not X_list:
        raise RuntimeError(f"No usable checkpoint data found in '{path}'.")

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    print(f"Pooled dataset: {len(X)} point(s), dim={X.shape[1]}")

    if len(X) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_points, replace=False)
        X, Y = X[idx], Y[idx]
        print(f"Subsampled to {max_points} point(s) (--max-points) to keep the "
              f"O(n^2)-O(n^3) kernel fit tractable.")

    return X, Y, param_names


# ---------------------------------------------------------------------------
# Part 2a: fresh full-rank ARD GP fit -> D = diag(1/lengthscale)
# ---------------------------------------------------------------------------

def fit_full_rank_gp(X, Y, x_min, kernel_type):
    """Fit the SAME bare (no ScaleKernel -- see get_covar_module_with_dim_scaled_prior's
    docstring, it returns a bare Matern/RBF with no separate outputscale, matching
    both botorch_optimize.py's actual GP and the user's k(x,y)=phi(D(x-y)) with no
    leading scale factor) ARD kernel used in this project, to obtain D=diag(1/lengthscale).
    The full-rank NLL reported here is then recomputed with the SAME formula used
    for the low-rank fit (U = I_d, i.e. k=d) so the two are directly comparable."""
    d = X.shape[1]
    X_norm = (X - x_min) / (1.0 - 2.0 * x_min)
    y_mean, y_std = Y.mean(), Y.std()
    Y_std = (Y - y_mean) / y_std

    X_t = torch.as_tensor(X_norm)
    Y_t = torch.as_tensor(Y_std).unsqueeze(-1)

    model = SingleTaskGP(
        X_t, Y_t,
        covar_module=get_covar_module_with_dim_scaled_prior(
            ard_num_dims=d, use_rbf_kernel=(kernel_type == "rbf")
        ),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

    with torch.no_grad():
        lengthscale = model.covar_module.lengthscale.detach().flatten().clone()
        noise = model.likelihood.noise.detach().flatten()[0].clone()

    D = 1.0 / lengthscale
    y_t = Y_t.squeeze(-1)
    theta_full = torch.cat([torch.eye(d, dtype=X_t.dtype).reshape(-1),
                             torch.log(noise).reshape(1)])
    with torch.no_grad():
        nll = _neg_log_marginal_likelihood(theta_full, X_t, y_t, D, d, d, kernel_type)

    info = {
        "lengthscale": lengthscale.tolist(),
        "noise": float(noise),
        "nll": float(nll),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }
    print(f"Stage A (full-rank GP): noise={info['noise']:.4e}  NLL={info['nll']:.4f}")
    return D, X_t, y_t, info


# ---------------------------------------------------------------------------
# Part 2b: low-rank kernel via Stiefel-constrained Newton-KKT
# ---------------------------------------------------------------------------

def _unpack(theta, d, k):
    U = theta[: d * k].reshape(d, k)
    log_noise = theta[d * k]
    return U, log_noise


def _kernel_matrix(theta, X, D, d, k, kernel_type):
    """K = phi(||U^T D (x_i - x_j)||) + noise * I -- no separate outputscale,
    matching the bare (unscaled) kernel this project actually uses (see
    fit_full_rank_gp's docstring) and the equation as given."""
    U, log_noise = _unpack(theta, d, k)
    n = X.shape[0]

    diff = X.unsqueeze(1) - X.unsqueeze(0)          # n x n x d
    proj = (diff * D) @ U                            # n x n x k
    # torch.norm's gradient (and especially its 2nd derivative) is NaN
    # exactly at 0, which happens on every diagonal entry (x_i - x_i = 0)
    # regardless of U -- an eps-regularized norm keeps the Hessian finite
    # there without changing phi(r) at any off-diagonal (r > 0) entry.
    r = torch.sqrt((proj ** 2).sum(dim=-1) + 1e-12)   # n x n

    if kernel_type == "rbf":
        phi = torch.exp(-0.5 * r ** 2)
    else:
        sqrt5 = 5.0 ** 0.5
        phi = (1.0 + sqrt5 * r + (5.0 / 3.0) * r ** 2) * torch.exp(-sqrt5 * r)

    noise = torch.exp(log_noise)
    jitter = 1e-6
    return phi + (noise + jitter) * torch.eye(n, dtype=X.dtype)


def _neg_log_marginal_likelihood(theta, X, y, D, d, k, kernel_type):
    n = X.shape[0]
    K = _kernel_matrix(theta, X, D, d, k, kernel_type)
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y.reshape(-1, 1), L)
    nll = (
        0.5 * (y.reshape(1, -1) @ alpha).squeeze()
        + torch.log(torch.diagonal(L)).sum()
        + 0.5 * n * float(np.log(2 * np.pi))
    )
    return nll


def loo_predictions(theta, X, y, D, d, k, kernel_type):
    """Closed-form leave-one-out predictive mean/variance (Rasmussen &
    Williams, eq. 5.12-5.13) -- no retraining, just one Cholesky/inverse of
    the already-fit kernel matrix. Used to compare how well the full-rank vs
    low-rank kernel actually predicts held-out points."""
    with torch.no_grad():
        K = _kernel_matrix(theta, X, D, d, k, kernel_type)
        L = torch.linalg.cholesky(K)
        K_inv = torch.cholesky_inverse(L)
        alpha = K_inv @ y
        k_inv_diag = torch.diagonal(K_inv)
        mu_loo = y - alpha / k_inv_diag
        var_loo = 1.0 / k_inv_diag
    return mu_loo, var_loo


def _constraint(U, k):
    """vec of the upper triangle (incl. diagonal) of U^T U - I_k."""
    M = U.T @ U - torch.eye(k, dtype=U.dtype)
    iu = torch.triu_indices(k, k)
    return M[iu[0], iu[1]]


def _polar_retract(U):
    """Project U onto the Stiefel manifold (nearest orthonormal matrix)."""
    Uu, _, Vt = torch.linalg.svd(U, full_matrices=False)
    return Uu @ Vt


def fit_low_rank_kernel_newton(X, y, D, k, kernel_type, n_iters, tol, seed):
    d = X.shape[1]
    m = k * (k + 1) // 2
    P = d * k + 1

    g_rng = torch.Generator().manual_seed(seed)
    U0, _ = torch.linalg.qr(torch.randn(d, k, generator=g_rng, dtype=X.dtype))
    theta = torch.cat([
        U0.reshape(-1),
        torch.zeros(1, dtype=X.dtype),   # log_noise = 0 -> noise = 1 (Y is standardized)
    ])

    def nll_fn(th):
        return _neg_log_marginal_likelihood(th, X, y, D, d, k, kernel_type)

    trace = []
    damping = 1e-4
    for it in range(n_iters):
        theta = theta.detach().requires_grad_(True)
        nll = nll_fn(theta)
        (g,) = torch.autograd.grad(nll, theta, create_graph=True)
        H = torch.autograd.functional.hessian(nll_fn, theta.detach())
        g = g.detach()

        U_cur = theta[: d * k].detach().reshape(d, k)
        c = _constraint(U_cur, k)
        J_U = torch.autograd.functional.jacobian(
            lambda u: _constraint(u.reshape(d, k), k), U_cur.reshape(-1)
        )
        J = torch.zeros(m, P, dtype=X.dtype)
        J[:, : d * k] = J_U

        grad_norm = g.norm().item()
        c_norm = c.norm().item()
        print(f"[newton {it:02d}] NLL={nll.item():.4f}  |g|={grad_norm:.3e}  "
              f"|c|={c_norm:.3e}  damping={damping:.1e}")
        trace.append({"iter": it, "nll": float(nll.item()), "grad_norm": grad_norm,
                       "constraint_violation": c_norm, "damping": damping})

        if grad_norm < tol and c_norm < tol:
            print("Converged (gradient + constraint tolerance).")
            break

        accepted = False
        cur_nll = nll.item()
        theta_flat = theta.detach()
        for _attempt in range(10):
            H_damped = H + damping * torch.diag(H.diagonal().abs() + 1.0)
            KKT = torch.zeros(P + m, P + m, dtype=X.dtype)
            KKT[:P, :P] = H_damped
            KKT[:P, P:] = J.T
            KKT[P:, :P] = J
            rhs = torch.cat([-g, -c])
            try:
                sol = torch.linalg.solve(KKT, rhs)
            except RuntimeError:
                damping *= 10
                continue
            if not torch.isfinite(sol).all():
                damping *= 10
                continue
            dtheta = sol[:P]

            # Geometric backtracking: the raw KKT/Newton direction can be far
            # too large when H is ill-conditioned or has negative curvature,
            # so try many halvings rather than stopping at a fixed floor.
            step = 1.0
            for _ in range(30):
                cand = theta_flat + step * dtheta
                if torch.isfinite(cand).all():
                    cand_U = _polar_retract(cand[: d * k].reshape(d, k))
                    if torch.isfinite(cand_U).all():
                        cand = torch.cat([cand_U.reshape(-1), cand[d * k:]])
                        with torch.no_grad():
                            cand_nll = nll_fn(cand).item()
                        if np.isfinite(cand_nll) and cand_nll < cur_nll:
                            theta = cand
                            accepted = True
                            break
                step *= 0.5
            if accepted:
                damping = max(damping / 2, 1e-8)
                break
            damping *= 4

        if not accepted:
            print("  no Newton step decreased NLL; stopping.")
            theta = theta_flat
            break

    U_final, log_noise = _unpack(theta.detach(), d, k)
    U_final = _polar_retract(U_final)
    final_nll = float(nll_fn(torch.cat([U_final.reshape(-1), log_noise.reshape(1)])).item())
    result = {
        "U": U_final,
        "noise": float(torch.exp(log_noise)),
        "nll": final_nll,
        "trace": trace,
    }
    print(f"Stage B (rank-{k} kernel): noise={result['noise']:.4e}  NLL={result['nll']:.4f}")
    return result


def plot_loo_comparison(actual, pred_full, pred_low, dim, rank, output_path):
    """Two-panel LOO predicted-vs-actual scatter (full-rank vs rank-k
    kernel), the most direct check of whether the low-rank subspace
    represents the data about as well as the full kernel: points hugging the
    y=x line means good predictions, points scattered off it mean the
    kernel (at that rank) doesn't explain those points' loss."""
    blue, muted, grid, text_secondary = "#2a78d6", "#898781", "#e1e0d9", "#52514e"

    lo = min(actual.min(), pred_full.min(), pred_low.min())
    hi = max(actual.max(), pred_full.max(), pred_low.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lims = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    stats = {}
    panels = [("full_rank", pred_full, f"Full-rank kernel (d={dim})"),
              ("low_rank", pred_low, f"Rank-{rank} kernel")]
    for ax, (key, pred, title) in zip(axes, panels):
        ax.scatter(actual, pred, s=18, color=blue, alpha=0.75, edgecolors="none")
        ax.plot(lims, lims, color=muted, linewidth=1, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("actual loss")
        rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
        ss_res = float(np.sum((pred - actual) ** 2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        stats[key] = {"loo_r2": r2, "loo_rmse": rmse}
        ax.set_title(f"{title}\nLOO R²={r2:.3f}  RMSE={rmse:.4f}", fontsize=10, color=text_secondary)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(grid)

    axes[0].set_ylabel("LOO-predicted loss")
    fig.suptitle("Leave-one-out cross-validated predictions (closed-form, no retraining)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return stats


def plot_projection_clusters(X, Y, D, U, output_path):
    """Project the pooled data onto the fitted rank-k subspace,
    z_i = U^T D (x_i - mean(X)), and scatter it colored by loss. This is a
    direct visual check of clustering/structure: if points close together in
    the low-rank space tend to share similar loss (color), the subspace is
    organizing the objective; if same-colored points are scattered all over,
    it isn't. For k>=3, a 3D view alone is hard to read from one fixed angle
    (occlusion/depth ambiguity in a static PNG), so the three pairwise 2D
    projections are plotted alongside it."""
    k = U.shape[1]
    Z = ((X - X.mean(axis=0)) * D) @ U   # n x k

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#104281", "#0d366b"]
    )
    grid = "#e1e0d9"

    if k == 1:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        sc = ax.scatter(Z[:, 0], Y, c=Y, cmap=cmap, s=20, edgecolors="none")
        ax.set_xlabel("u0")
        ax.set_ylabel("loss")
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        fig.colorbar(sc, ax=ax, label="loss")
        fig.suptitle("Rank-1 projection vs. loss")
    elif k == 2:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=Y, cmap=cmap, s=20, edgecolors="none")
        ax.set_xlabel("u0")
        ax.set_ylabel("u1")
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        fig.colorbar(sc, ax=ax, label="loss")
        fig.suptitle("Rank-2 projection, colored by loss")
    else:
        fig = plt.figure(figsize=(11, 9), constrained_layout=True)
        ax3d = fig.add_subplot(2, 2, 1, projection="3d")
        sc = ax3d.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=Y, cmap=cmap, s=16, edgecolors="none")
        ax3d.set_xlabel("u0")
        ax3d.set_ylabel("u1")
        ax3d.set_zlabel("u2")
        ax3d.set_title("3D view (u0, u1, u2)", fontsize=10)

        for i, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)], start=2):
            ax = fig.add_subplot(2, 2, i)
            ax.scatter(Z[:, a], Z[:, b], c=Y, cmap=cmap, s=16, edgecolors="none")
            ax.set_xlabel(f"u{a}")
            ax.set_ylabel(f"u{b}")
            ax.grid(True, color=grid, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_title(f"u{a} vs u{b}", fontsize=10)

        fig.colorbar(sc, ax=fig.get_axes(), label="loss", shrink=0.6)
        fig.suptitle(f"Data projected onto the fitted rank-{k} subspace, colored by loss", fontsize=12)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Part 3: main -- tie the three parts together, save + print results
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="garabedian_linear_cei_pressure", help="wandb project name")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="wandb entity (team/user)")
    parser.add_argument("--config-path", default="../configs/garabedian_linear_pressure.yaml")
    parser.add_argument("--rank", type=int, default=3, help="columns of U (target subspace dimension k)")
    parser.add_argument("--max-points", type=int, default=300,
                         help="subsample the pooled dataset to at most this many points")
    parser.add_argument("--n-newton-iters", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="low_rank_kernel_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("wandb_project") != args.project:
        print(f"WARNING: --config-path's wandb_project ('{cfg.get('wandb_project')}') "
              f"does not match --project ('{args.project}'). Double check --config-path.")
    kernel_type = cfg.get("kernel", "matern")
    x_min = float(cfg["x_min"])

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # --- Part 1 ---
    api = wandb.Api()
    X, Y, param_names = gather_data(
        api, args.entity, args.project, cfg["use_gp_constraints"], args.max_points, args.random_seed
    )

    with open(os.path.join(args.output_dir, "pooled_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(param_names) + ["loss"])
        for x_row, y_val in zip(X, Y):
            w.writerow(list(x_row) + [y_val])

    # --- Part 2 ---
    D, X_t, y_t, stage_a_info = fit_full_rank_gp(X, Y, x_min, kernel_type)
    result = fit_low_rank_kernel_newton(
        X_t, y_t, D, args.rank, kernel_type, args.n_newton_iters, args.tol, args.random_seed
    )

    # --- Part 3 ---
    d = X_t.shape[1]
    theta_full = torch.cat([
        torch.eye(d, dtype=X_t.dtype).reshape(-1),
        torch.log(torch.tensor(stage_a_info["noise"], dtype=X_t.dtype)).reshape(1),
    ])
    theta_low = torch.cat([result["U"].reshape(-1), torch.log(torch.tensor(result["noise"], dtype=X_t.dtype)).reshape(1)])
    mu_loo_full_std, _ = loo_predictions(theta_full, X_t, y_t, D, d, d, kernel_type)
    mu_loo_low_std, _ = loo_predictions(theta_low, X_t, y_t, D, d, args.rank, kernel_type)

    y_mean, y_std = stage_a_info["y_mean"], stage_a_info["y_std"]
    pred_full = mu_loo_full_std.numpy() * y_std + y_mean
    pred_low = mu_loo_low_std.numpy() * y_std + y_mean
    plot_path = os.path.join(args.output_dir, "loo_predicted_vs_actual.png")
    loo_stats = plot_loo_comparison(Y, pred_full, pred_low, d, args.rank, plot_path)
    print(f"LOO full-rank:  R²={loo_stats['full_rank']['loo_r2']:.3f}  "
          f"RMSE={loo_stats['full_rank']['loo_rmse']:.4f}")
    print(f"LOO rank-{args.rank}: R²={loo_stats['low_rank']['loo_r2']:.3f}  "
          f"RMSE={loo_stats['low_rank']['loo_rmse']:.4f}")

    cluster_plot_path = os.path.join(args.output_dir, "projection_clusters.png")
    plot_projection_clusters(X, Y, D.numpy(), result["U"].numpy(), cluster_plot_path)

    U_path = os.path.join(args.output_dir, "U_matrix.csv")
    with open(U_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param_name"] + [f"u{j}" for j in range(args.rank)])
        for name, urow in zip(param_names, result["U"].tolist()):
            w.writerow([name] + urow)
    print(f"Wrote {U_path}")

    trace_path = os.path.join(args.output_dir, "newton_trace.csv")
    with open(trace_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "nll", "grad_norm", "constraint_violation", "damping"])
        w.writeheader()
        w.writerows(result["trace"])
    print(f"Wrote {trace_path}")

    summary = {
        "project": args.project,
        "n_points": len(X),
        "dim": X.shape[1],
        "rank": args.rank,
        "kernel_type": kernel_type,
        "D": D.tolist(),
        "stage_a_full_rank": stage_a_info,
        "stage_b_low_rank": {
            "noise": result["noise"],
            "nll": result["nll"],
            "n_newton_iters_run": len(result["trace"]),
        },
        "nll_gap": result["nll"] - stage_a_info["nll"],
        "loo": loo_stats,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    print(f"\nFull-rank NLL = {stage_a_info['nll']:.4f}   "
          f"rank-{args.rank} NLL = {result['nll']:.4f}   "
          f"gap = {summary['nll_gap']:.4f} (0 = no loss from the low-rank compression; "
          f"lower gap = the rank-{args.rank} subspace explains the data about as well as the full kernel).")


if __name__ == "__main__":
    main()
