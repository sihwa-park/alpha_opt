#!/usr/bin/env python
"""
Render the single optimization frame with the lowest loss as a PDF (same 4-panel
layout as make_videos.py, but one vector image instead of a video).

Usage:
    python make_best_frame.py <runname>

Example:
    python make_best_frame.py divine-firebrand-74
"""

import argparse
import os
import shutil
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the sibling module directly by path, bypassing the `alpha_opt` package
# __init__ (which pulls in firm3d/firm3dpp — unneeded here and currently crashes
# with "Illegal instruction" on some environments).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_videos import (  # noqa: E402
    DEFAULT_DPI,
    DEFAULT_EVALS_DIR,
    build_frame_figure,
    fetch_wandb_history,
)


def find_best_index(history_df):
    """Return the iteration index with the lowest loss among non-failed evals."""
    df = history_df
    if "vmec_failed" in df.columns:
        df = df[df["vmec_failed"] <= 0.5]
    if df.empty:
        raise ValueError("No successful (non-VMEC-failed) evaluations in wandb history")
    best_row = df.loc[df["loss"].idxmin()]
    return int(best_row["iteration"]), float(best_row["loss"])


def main():
    parser = argparse.ArgumentParser(description="Save the best-loss optimization frame as a PDF.")
    parser.add_argument("project_name", help="Project name, e.g. garabedian_linear_cei")
    parser.add_argument("run_name", help="Run name, e.g. divine-firebrand-74")
    parser.add_argument("--evals-dir", default=DEFAULT_EVALS_DIR)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args()

    print("Fetching wandb history...", flush=True)
    history_df = fetch_wandb_history(args.project_name, args.run_name)
    if history_df is None:
        raise RuntimeError(f"No wandb history found for run '{args.run_name}'")

    best_idx, best_loss = find_best_index(history_df)
    print(f"Best loss = {best_loss:.6g} at iteration {best_idx}", flush=True)

    eval_dir = os.path.join(args.evals_dir, f"{args.run_name}_eval{best_idx:06d}")
    wout_file = os.path.join(eval_dir, "wout_tmp.nc")
    particle_csv = os.path.join(eval_dir, "particle_data.csv")
    label = f"{args.run_name}  |  best eval {best_idx:06d}  |  loss={best_loss:.6g}"

    tmpdir = tempfile.mkdtemp()
    try:
        fig = build_frame_figure(wout_file, particle_csv, label, tmpdir,
                                  history_df=history_df, current_idx=best_idx)

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{args.run_name}_best_eval{best_idx:06d}.png")
        print(f"Saving {output_path}...", flush=True)
        fig.savefig(output_path, dpi=args.dpi)
        plt.close(fig)
        print("Done.", flush=True)
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
