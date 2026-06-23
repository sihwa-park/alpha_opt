#!/usr/bin/env python
"""
Reproduce trial 16 and check stz_inits / grid indices before the GPU call.

Run from the alpha_opt root:
    python experiments/debug_trial16.py [path/to/config.yaml]
"""

import os
import sys
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Hardcoded x from the failing trial 16
# ---------------------------------------------------------------------------
# X = np.array([
#     0.052528, 0.343611, 0.886642, 0.880989, 0.016367, 0.820174, 0.7586,   0.937056,
#     0.050093, 0.448216, 0.671892, 0.156337, 0.394539, 0.484164, 0.486364, 0.672674,
#     0.928225, 0.822357, 0.417637, 0.213327, 0.200875, 0.892966, 0.194755,
# ])
X = np.array([
    0.862138, 0.345147, 0.753571, 0.730658, 0.704828, 0.180403, 0.106588, 0.228497,
    0.481859, 0.616802, 0.330428, 0.345718, 0.32783,  0.591369, 0.286581, 0.341244,
    0.77426,  0.532905, 0.521523, 0.374228, 0.478494, 0.337503, 0.763703,
])


def check_grid_indices(stz_inits, srange, trange, zrange, field=None):
    """Replicate build_state index math and report any out-of-bounds indices."""
    s     = stz_inits[:, 0].copy()
    theta = stz_inits[:, 1].copy()
    zeta  = stz_inits[:, 2].copy()

    print("\n=== Grid index diagnostics ===")
    print(f"srange:  [{srange[0]:.4f}, {srange[1]:.4f}], n={int(srange[2])}")
    print(f"trange:  [{trange[0]:.4f}, {trange[1]:.4f}], n={int(trange[2])}")
    print(f"zrange:  [{zrange[0]:.4f}, {zrange[1]:.4f}], n={int(zrange[2])}")
    print()

    # NaN/Inf in raw inputs
    print(f"NaN in s:     {np.isnan(s).sum()},  Inf: {np.isinf(s).sum()}")
    print(f"NaN in theta: {np.isnan(theta).sum()},  Inf: {np.isinf(theta).sum()}")
    print(f"NaN in zeta:  {np.isnan(zeta).sum()},  Inf: {np.isinf(zeta).sum()}")
    print()
    print(f"s  values: min={np.nanmin(s):.6f}, max={np.nanmax(s):.6f}")
    print(f"theta:     min={np.nanmin(theta):.6f}, max={np.nanmax(theta):.6f}")
    print(f"zeta:      min={np.nanmin(zeta):.6f}, max={np.nanmax(zeta):.6f}")
    print()

    # map_to_grid (Boozer) — same logic as C++ kernel
    t = np.mod(theta, 2 * np.pi)
    period = zrange[1]
    z = np.mod(zeta, period)

    sym = t > np.pi
    z[sym] = period - z[sym]
    t[sym] = 2 * np.pi - t[sym]

    x1, x2, x3 = s, t, z

    print(f"x1 (s after map):  min={np.nanmin(x1):.6f}, max={np.nanmax(x1):.6f},  NaN={np.isnan(x1).sum()}")
    print(f"x2 (t after map):  min={np.nanmin(x2):.6f}, max={np.nanmax(x2):.6f},  NaN={np.isnan(x2).sum()}")
    print(f"x3 (z after map):  min={np.nanmin(x3):.6f}, max={np.nanmax(x3):.6f},  NaN={np.isnan(x3).sum()}")
    print()

    # grid step sizes
    x1_gs = (srange[1] - srange[0]) / (srange[2] - 1)
    x2_gs = (trange[1] - trange[0]) / (trange[2] - 1)
    x3_gs = (zrange[1] - zrange[0]) / (zrange[2] - 1)

    # raw indices (before clamping) — matches build_state C++ exactly
    # NaN inputs produce 0 from astype(int) in numpy (unlike C++ where it's UB)
    i_raw = (3 * ((x1 - srange[0]) / x1_gs).astype(int) // 3) * 3
    j_raw = (3 * ((x2 - trange[0]) / x2_gs).astype(int) // 3) * 3
    k_raw = (3 * ((x3 - zrange[0]) / x3_gs).astype(int) // 3) * 3

    print(f"i_raw: min={i_raw.min()}, max={i_raw.max()}")
    print(f"j_raw: min={j_raw.min()}, max={j_raw.max()}  <-- negative = BUG")
    print(f"k_raw: min={k_raw.min()}, max={k_raw.max()}  <-- negative = BUG")
    print()
    print(f"particles with s  >= 1.0:   {(s  >= 1.0).sum()}")
    print(f"particles with i_raw < 0:   {(i_raw < 0).sum()}")
    print(f"particles with j_raw < 0:   {(j_raw < 0).sum()}")
    print(f"particles with k_raw < 0:   {(k_raw < 0).sum()}")

    n_bad = ((i_raw < 0) | (j_raw < 0) | (k_raw < 0) | (s >= 1.0)).sum()
    print(f"\nTotal particles with any OOB issue: {n_bad} / {len(s)}")

    if field is not None:
        print("\n=== Field values at initial positions (sample of 1000) ===")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(stz_inits), min(1000, len(stz_inits)), replace=False)
        field.set_points(stz_inits[idx])
        modB = field.modB().flatten()
        G    = field.G().flatten()
        iota = field.iota().flatten()
        psi0 = field.psi0
        print(f"psi0  = {psi0:.6f}")
        print(f"modB:  min={modB.min():.4f},  max={modB.max():.4f},  NaN={np.isnan(modB).sum()}")
        print(f"G:     min={G.min():.4f},  max={G.max():.4f},  NaN={np.isnan(G).sum()}")
        print(f"iota:  min={iota.min():.4f},  max={iota.max():.4f},  NaN={np.isnan(iota).sum()}")
        print(f"G/modB (dtmax scale): min={(G/modB).min():.4f},  max={(G/modB).max():.4f}")
        print(f"dmodBdpsi scale (modB range / psi0): {(modB.max()-modB.min())/abs(psi0):.6f}")
        print("=========================================================")

    print("==============================\n")


def make_patched_gpu(real_fn, field_holder=None):
    """Return a wrapper around real_fn that prints diagnostics first."""
    def patched(**kwargs):
        stz = kwargs["stz_init"]
        sr  = kwargs["srange"]
        tr  = kwargs["trange"]
        zr  = kwargs["zrange"]

        field = field_holder[0] if field_holder else None
        check_grid_indices(np.array(stz), tuple(sr), tuple(tr), tuple(zr), field=field)

        print("Calling real boozer_gpu_tracing (set SKIP_GPU=1 to abort)...")
        if os.environ.get("SKIP_GPU") == "1":
            print("SKIP_GPU=1 — aborting before GPU call.")
            sys.exit(0)

        return real_fn(**kwargs)
    return patched


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "alpha_opt/configs/garabedian_qt.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for k in ["t_max", "tau", "tol", "max_B_target", "x_min"]:
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])

    # --- Geometry ---
    aspect_ratio = cfg["aspect_ratio"]
    minor_radius = (
        float(cfg["minor_radius"])
        if cfg.get("minor_radius") is not None
        else 3.1 / aspect_ratio ** 0.38
    )
    major_radius = minor_radius * aspect_ratio

    # --- VMEC ---
    from vmecpp.simsopt_compat import Vmec
    vmec = Vmec(cfg["vmec_input_file"], verbose=True)
    avg_B_estimate = cfg["max_B_target"] / np.sqrt(2)
    phiedge_high = np.pi * avg_B_estimate * minor_radius ** 2 * 2
    vmec.set("phiedge", phiedge_high)

    # --- Surface ---
    from alpha_opt.surface import SurfaceGarabedianQuantiles, SurfaceGarabedianLinear
    if cfg["parameterization"] == "garabedian_quantiles":
        surface = SurfaceGarabedianQuantiles(
            vmec.indata.nfp,
            mpol=cfg["mpol"],
            ntor=cfg["ntor"],
            minor_radius=minor_radius,
            major_radius=major_radius,
            filename=cfg["data_file"],
            exact_radii=True,
        )
    elif cfg["parameterization"] == "garabedian_linear":
        surface = SurfaceGarabedianLinear(
            vmec.indata.nfp,
            mpol=cfg["mpol"],
            ntor=cfg["ntor"],
            minor_radius=minor_radius,
            major_radius=major_radius,
            filename=cfg["data_file"],
            exact_radii=True,
        )
    else:
        raise ValueError(f"Unsupported parameterization: {cfg['parameterization']}")

    # --- Objective (with monkey-patched GPU call) ---
    import alpha_opt.tracing.alpha_tracing as _at

    # Capture the field object after generate_interpolant_and_initial_conditions runs
    field_holder = [None]
    _orig_gen = _at.generate_interpolant_and_initial_conditions
    def _capturing_gen(*args, **kwargs):
        result = _orig_gen(*args, **kwargs)
        field_holder[0] = result[-1]  # field is the last element of the returned tuple
        return result
    _at.generate_interpolant_and_initial_conditions = _capturing_gen

    _real_gpu = _at.firm3dpp.boozer_gpu_tracing
    _at.firm3dpp.boozer_gpu_tracing = make_patched_gpu(_real_gpu, field_holder)

    from alpha_opt.objective import get_objective
    x_scale = np.ones(len(surface.x))

    eval_dir = "debug_trial16_eval"
    os.makedirs(eval_dir, exist_ok=True)
    os.chdir(eval_dir)

    def raw_objective():
        from alpha_opt.tracing import compute_alpha_loss
        vmec.wout.save("wout_tmp.nc")
        return compute_alpha_loss(
            "wout_tmp.nc",
            n_particles=cfg["n_particles"],
            t_max=cfg["t_max"],
            tau=cfg["tau"],
            t_block=cfg["t_block"],
            min_dt=cfg["min_dt"],
            maxloss=cfg["maxloss"],
            tol=cfg["tol"],
            vacuum=cfg["vacuum"],
        )

    objective = get_objective(
        vmec,
        surface,
        x_scale,
        raw_objective,
        max_B=cfg["max_B_target"],
        max_B_iterations=cfg["max_B_iterations"],
        phiedge=phiedge_high,
    )

    print(f"Running objective with hardcoded x (dim={len(X)})...")
    try:
        objective(X)
    except SystemExit:
        pass  # raised by patched_boozer_gpu_tracing when SKIP_GPU=1
    except Exception as e:
        print(f"Exception: {e}")

    # Restore
    _at.firm3dpp.boozer_gpu_tracing = _real_gpu
    _at.generate_interpolant_and_initial_conditions = _orig_gen


if __name__ == "__main__":
    main()
