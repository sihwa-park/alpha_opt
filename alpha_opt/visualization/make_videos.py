#!/usr/bin/env python
"""
Generate a single optimization visualization video with 4 subplots per frame.

Layout per frame:
  top-left:     |B| Boozer contours (2x2 grid for rho=0.25,0.5,0.75,1.0)
  top-right:    VMEC 3D surface
  bottom-left:  VMEC cross-sections
  bottom-right: Alpha particle loss curves

Usage:
    python make_videos.py <runname> <start_index> <end_index>

Example:
    python make_videos.py divine-firebrand-74 0 50
"""

import argparse
import io
import os
import shutil
import tempfile

import booz_xform as bx
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.colors import LightSource
import numpy as np
import pandas as pd
import scipy
from scipy.io import netcdf_file
import textwrap
import wandb

DEFAULT_EVALS_DIR = os.path.join(os.path.expanduser("~"), "evals")
DEFAULT_FPS = 4
DEFAULT_DPI = 150

def fetch_wandb_history(project, runname, entity="sp2582-cornell-university"):
    """Fetch full loss history from wandb for a given run name."""
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"display_name": runname})
    if not runs:
        print(f"Warning: no wandb run found for '{runname}'")
        return None
    run = runs[0]
    history = run.scan_history(keys=["iteration", "loss", "case", "vmec_failed"])
    df = pd.DataFrame(history)
    if df.empty:
        return None
    df = df.sort_values("iteration").reset_index(drop=True)
    return df


def fig_to_array(fig, dpi=DEFAULT_DPI):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return imageio.v3.imread(buf)

def draw_wandb_loss(ax, history_df, current_idx):
    """Plot full loss history and highlight the current iteration's dot."""
    if history_df is None or history_df.empty:
        draw_vmec_failed(ax, "No wandb data")
        return

    # Use full history for plotting
    iters = history_df["iteration"].values
    loss = history_df["loss"].values
    vmec_failed = history_df["vmec_failed"].values if "vmec_failed" in history_df.columns else None

    # Mark failed points
    if vmec_failed is not None:
        failed_mask = vmec_failed > 0.5
        ax.scatter(iters[failed_mask], loss[failed_mask],
                   color="red", s=20, zorder=3, label="VMEC failed", marker="x")
        ax.scatter(iters[~failed_mask], loss[~failed_mask],
                   color="steelblue", s=10, zorder=3, alpha=0.6)
    else:
        ax.scatter(iters, loss, color="steelblue", s=10, zorder=3, alpha=0.6)

    ax.plot(iters, loss, color="steelblue", linewidth=0.8, alpha=0.4)

    # Highlight only the current iteration dot
    cur = history_df[history_df["iteration"] == current_idx]
    if not cur.empty:
        ax.scatter(cur["iteration"].values, cur["loss"].values,
                   color="yellow", s=80, zorder=5, edgecolors="black", linewidths=1.2,
                   label=f"Current (iter {current_idx})")

    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel("Loss", fontsize=8)
    ax.set_title("Optimization loss", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_ylim(-1.0, 5.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_case_history(ax, history_df, current_idx):
    """Plot full case history and highlight the current iteration."""
    if history_df is None or history_df.empty or "case" not in history_df.columns:
        draw_vmec_failed(ax, "No case data")
        return

    iters = history_df["iteration"].values
    case = history_df["case"].values

    ax.scatter(iters, case, color="steelblue", s=10, zorder=3, alpha=0.6)
    ax.plot(iters, case, color="steelblue", linewidth=0.8, alpha=0.4)

    # Highlight current iteration
    cur = history_df[history_df["iteration"] == current_idx]
    if not cur.empty:
        ax.scatter(cur["iteration"].values, cur["case"].values,
                   color="yellow", s=80, zorder=5, edgecolors="black", linewidths=1.2,
                   label=f"Current (iter {current_idx})")

    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel("Case", fontsize=8)
    ax.set_yticks([-1, 0, 1])
    ax.set_title("Case history", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_vmec_failed(ax, message="VMEC Failed"):
    wrapped = "\n".join(textwrap.wrap(message, width=40))
    if hasattr(ax, 'text2D'):  # Axes3D
        ax.text2D(0.5, 0.5, wrapped, ha="center", va="center",
                  fontsize=14, color="red", transform=ax.transAxes)
    else:  # regular 2D Axes
        ax.text(0.5, 0.5, wrapped, ha="center", va="center",
                fontsize=14, color="red", transform=ax.transAxes)
    ax.axis("off")
    ax.axis("off")


def draw_boozer(boozer_axes, wout_filename, tmpdir):
    rho_values = [0.25, 0.5, 0.75, 1.0]
    s_targets = [rho**2 for rho in rho_values]

    b = bx.Booz_xform()
    b.read_wout(wout_filename)
    ns_in = b.ns_in
    s_half = np.array([(j + 0.5) / ns_in for j in range(ns_in)])
    compute_surfs = [int(np.argmin(np.abs(s_half - s_t))) for s_t in s_targets]
    b.mboz = 24
    b.nboz = 24
    b.compute_surfs = compute_surfs
    b.run()

    boozmn_file = os.path.join(tmpdir, "boozmn_tmp.nc")
    b.write_boozmn(boozmn_file)

    with netcdf_file(boozmn_file, mmap=False) as f:
        jlist = f.variables["jlist"][()].copy()
        iota_b = f.variables["iota_b"][()].copy()
        xm = f.variables["ixm_b"][()].copy()
        xn = f.variables["ixn_b"][()].copy()
        bmnc = f.variables["bmnc_b"][()].copy()

    ntheta, nphi = 70, 80
    theta1d = np.linspace(0, 2 * np.pi, ntheta)
    phi1d = np.linspace(0, 2 * np.pi / 2, nphi)
    phi2d, theta2d = np.meshgrid(phi1d, theta1d)
    rounded_rhos = [0.25, 0.5, 0.75, 1]

    for js, ax in enumerate(boozer_axes):
        iota = iota_b[jlist[js] - 2]
        modB = np.zeros((ntheta, nphi))
        for jmn in range(len(xm)):
            modB += bmnc[js, jmn] * np.cos(xm[jmn] * theta2d - xn[jmn] * phi2d)

        cp = ax.contour(phi1d, theta1d, modB, 20, linewidths=0.8)
        ax.plot([phi1d[0], phi1d[-1]], [iota * phi1d[0], iota * phi1d[-1]], "k-", linewidth=1.0)
        plt.colorbar(cp, ax=ax, pad=0.02)
        ax.tick_params(direction="in", length=0, labelsize=6)
        ax.set_title(rf"|B| [T], $\rho$={rounded_rhos[js]}", fontsize=8)
        ax.set_yticks([0, 2 * np.pi])
        ax.set_yticklabels(["0", r"$2\pi$"], fontsize=6)
        ax.set_xticks([0, np.pi])
        ax.set_xticklabels(["0", r"$\pi$"], fontsize=6)
        ax.set_ylabel(r"$\theta_B$", labelpad=-8, fontsize=7)
        ax.set_xlabel(r"$\varphi_B$", labelpad=-4, fontsize=7)
        ax.set_box_aspect(1)


def draw_vmec3d(ax, wout_filename):
    with netcdf_file(wout_filename, "r", mmap=False) as f:
        ns = f.variables["ns"][()]
        xn = f.variables["xn"][()]
        xm = f.variables["xm"][()]
        xn_nyq = f.variables["xn_nyq"][()]
        xm_nyq = f.variables["xm_nyq"][()]
        rmnc = f.variables["rmnc"][()]
        zmns = f.variables["zmns"][()]
        bmnc = f.variables["bmnc"][()]
        Rmajor = f.variables["Rmajor_p"][()]

    ntheta, nphi = 90, 500
    theta1D = np.linspace(0, 2 * np.pi, num=ntheta)
    phi1D = np.linspace(0, 2 * np.pi, num=nphi)
    phi2D, theta2D = np.meshgrid(phi1D, theta1D)
    iradius = ns - 1

    angles = xm[:, None, None] * theta2D[None] - xn[:, None, None] * phi2D[None]
    R = np.sum(rmnc[iradius, :, None, None] * np.cos(angles), axis=0) / Rmajor
    Z = np.sum(zmns[iradius, :, None, None] * np.sin(angles), axis=0) / Rmajor

    angles_nyq = xm_nyq[:, None, None] * theta2D[None] - xn_nyq[:, None, None] * phi2D[None]
    B = np.sum(bmnc[iradius, :, None, None] * np.cos(angles_nyq), axis=0)

    X = R * np.cos(phi2D)
    Y = R * np.sin(phi2D)
    B_rescaled = (B - B.min()) / (B.max() - B.min())

    ax.set_axis_off()
    ls = LightSource(azdeg=45, altdeg=60)
    rgb = ls.shade(B_rescaled, cmap=cm.viridis, vert_exag=0.3, blend_mode="soft")
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, antialiased=True, shade=True)

    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0 * 0.8
    for setter, arr in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], [X, Y, Z]):
        mid = (arr.max() + arr.min()) / 2
        setter(mid - max_range, mid + max_range)
    ax.set_title("3D surface (color = |B|)", fontsize=9)


def draw_xsections(ax, wout_filename):
    with netcdf_file(wout_filename, "r", mmap=False) as f:
        ns = f.variables["ns"][()]
        nfp = f.variables["nfp"][()]
        xn = f.variables["xn"][()]
        xm = f.variables["xm"][()]
        rmnc = f.variables["rmnc"][()]
        zmns = f.variables["zmns"][()]
        lasym = f.variables["lasym__logical__"][()]
        if lasym == 1:
            rmns = f.variables["rmns"][()]
            zmnc = f.variables["zmnc"][()]
        else:
            rmns = np.zeros_like(rmnc)
            zmnc = np.zeros_like(rmnc)

    ntheta, nphi_plot = 200, 4
    theta = np.linspace(0, 2 * np.pi, num=ntheta)
    phi = np.linspace(0, 2 * np.pi / nfp, num=nphi_plot, endpoint=False)
    iradius = ns - 1

    angles = xm[None, None, :] * theta[:, None, None] - xn[None, None, :] * phi[None, :, None]
    R = np.sum(rmnc[iradius][None, None, :] * np.cos(angles) + rmns[iradius][None, None, :] * np.sin(angles), axis=-1)
    Z = np.sum(zmns[iradius][None, None, :] * np.sin(angles) + zmnc[iradius][None, None, :] * np.cos(angles), axis=-1)

    ax.plot(R[:, 0], Z[:, 0], "r", label=r"$\phi=0$")
    ax.plot(R[:, 1], Z[:, 1], "g", label="1/4 period")
    ax.plot(R[:, 2], Z[:, 2], "b", label="1/2 period")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("R [m]", fontsize=8)
    ax.set_ylabel("Z [m]", fontsize=8)
    ax.set_xlim(5.5, 13)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title("Cross-sections", fontsize=9)
    ax.tick_params(labelsize=7)


def draw_losses(ax, particle_csv):
    tau = 0.1
    maxloss = 0.02

    particle_data = pd.read_csv(particle_csv)
    last_times = particle_data["last_time"].values
    sorted_times = np.sort(last_times)
    n_particles = len(sorted_times)

    cumulative_particle_loss = np.arange(1, n_particles + 1) / n_particles
    energy_loss_per_particle = np.exp(-sorted_times / tau)
    cumulative_energy_loss = np.cumsum(energy_loss_per_particle) / n_particles

    idx = np.searchsorted(cumulative_energy_loss, maxloss, side="left")
    time_at_maxloss = sorted_times[idx] if idx < n_particles else None

    ax.plot(sorted_times, cumulative_particle_loss, "b-", linewidth=1.5, label="Particle loss")
    ax.plot(sorted_times, cumulative_energy_loss, "r-", linewidth=1.5, label=f"Energy loss (τ={tau})")
    ax.axhline(y=maxloss, color="k", linestyle="--", linewidth=1.2, label=f"Threshold ({maxloss})")
    if time_at_maxloss is not None:
        ax.axvline(x=time_at_maxloss, color="g", linestyle="--", linewidth=1.2,
                   label=f"t={time_at_maxloss:.2e} s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Cumulative loss fraction", fontsize=9)
    ax.set_title("Alpha particle losses", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)


def build_frame_figure(wout_file, particle_csv, label, tmpdir, history_df=None, current_idx=0):
    """Build and return the per-frame Figure (caller is responsible for closing it)."""
    fig = plt.figure(figsize=(16, 9.6))
    fig.suptitle(label, fontsize=11, y=0.99)

    outer = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                              left=0.05, right=0.97, top=0.95, bottom=0.06)

    # Top-left: wandb loss graph
    ax_wandb = fig.add_subplot(outer[0, 0])

    # Top-middle: 2x2 Boozer panels (tighter margins)
    inner_boozer = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 1],
                                                    hspace=0.25, wspace=0.25)
    boozer_axes = [fig.add_subplot(inner_boozer[r, c]) for r in range(2) for c in range(2)]

    # Top-right: 3D surface
    ax_3d = fig.add_subplot(outer[0, 2], projection="3d")

    # Bottom-left: cross-sections
    ax_xsec = fig.add_subplot(outer[1, 0])

    # Bottom-middle: losses
    ax_loss = fig.add_subplot(outer[1, 1])

    # Bottom-right: case history
    ax_case = fig.add_subplot(outer[1, 2])


    # Draw wandb loss (always attempted)
    draw_wandb_loss(ax_wandb, history_df, current_idx)
    draw_case_history(ax_case, history_df, current_idx)

    vmec_ok = os.path.exists(wout_file)

    if not vmec_ok:
        for ax in boozer_axes:
            draw_vmec_failed(ax, "VMEC Failed")
        draw_vmec_failed(ax_3d)
        draw_vmec_failed(ax_xsec)
        draw_vmec_failed(ax_loss)
        return fig

    try:
        draw_boozer(boozer_axes, wout_file, tmpdir)
    except Exception as e:
        print(f"  Boozer error: {e}", flush=True)
        for ax in boozer_axes:
            draw_vmec_failed(ax, f"Boozer error:\n{e}")

    try:
        draw_vmec3d(ax_3d, wout_file)
    except Exception as e:
        print(f"  3D error: {e}", flush=True)
        draw_vmec_failed(ax_3d, f"3D error:\n{e}")

    try:
        draw_xsections(ax_xsec, wout_file)
    except Exception as e:
        print(f"  Xsec error: {e}", flush=True)
        draw_vmec_failed(ax_xsec, f"Xsec error:\n{e}")

    if not os.path.exists(particle_csv):
        draw_vmec_failed(ax_loss, "No particle data")
    else:
        try:
            draw_losses(ax_loss, particle_csv)
        except Exception as e:
            print(f"  Loss error: {e}", flush=True)
            draw_vmec_failed(ax_loss, f"Loss error:\n{e}")

    return fig


def make_frame(wout_file, particle_csv, label, tmpdir, history_df=None, current_idx=0, dpi=DEFAULT_DPI):
    fig = build_frame_figure(wout_file, particle_csv, label, tmpdir, history_df, current_idx)
    arr = fig_to_array(fig, dpi)
    plt.close(fig)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Generate optimization visualization video.")
    parser.add_argument("project_name", help="Project name, e.g. garabedian_linear_cei")
    parser.add_argument("run_name", help="Run name, e.g. divine-firebrand-74")
    parser.add_argument("start_index", type=int, help="Start evaluation index (inclusive)")
    parser.add_argument("end_index", type=int, help="End evaluation index (inclusive)")
    parser.add_argument("--evals-dir", default=DEFAULT_EVALS_DIR)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args()
    # Fetch wandb history once upfront
    print("Fetching wandb history...", flush=True)
    history_df = fetch_wandb_history(args.run_name, args.project_name)

    tmpdir = tempfile.mkdtemp()
    frames = []
    try:
        for idx in range(args.start_index, args.end_index + 1):
            eval_dir = os.path.join(args.evals_dir, f"{args.run_name}_eval{idx:06d}")
            wout_file = os.path.join(eval_dir, "wout_tmp.nc")
            particle_csv = os.path.join(eval_dir, "particle_data.csv")
            label = f"{args.run_name}  |  eval {idx:06d}"

            print(f"[{idx:06d}] Rendering...", flush=True)
            frame = make_frame(wout_file, particle_csv, label, tmpdir,
                               history_df=history_df, current_idx=idx, dpi=args.dpi)
            frames.append(frame)

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"{args.run_name}_{args.start_index:06d}_{args.end_index:06d}.mp4",
        )
        print(f"Saving {output_path} ({len(frames)} frames)...", flush=True)
        imageio.mimwrite(output_path, frames, fps=args.fps)
        print("Done.", flush=True)

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
