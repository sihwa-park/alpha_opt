#!/usr/bin/env python
"""
Diagnose WHICH part of the magnetic configuration (for trial 16's hardcoded X)
produces the runaway `s` and the blown-up RHS denominator `D`.

This reuses the debug_trial16 harness (VMEC -> surface -> interpolant -> field),
but instead of launching the GPU kernel it INTERCEPTS the exact arguments that
would have been passed to boozer_gpu_tracing. That gives us the real
mass / charge / v_total / psi0 / initial vpar / grid ranges and the captured
field object -- so nothing about the physics constants is guessed.

It then:
  1. Reports VMEC equilibrium quality (aspect, beta, iota profile, axis).
  2. Scans modB, G, I, iota, K (+ derivatives) over the FULL (s, theta, zeta)
     grid -- not just initial particle positions -- looking for NaN/Inf,
     extreme values, and iota zero-crossings.
  3. Reconstructs the GC_Boozer denominator
        C = -m*vpar*dKdzeta/modB - q*iota + m*vpar*dGdpsi/modB
        F = -m*vpar*dKdtheta/modB + q     + m*vpar*dIdpsi/modB
        D = (F*G - C*I) / iota
     over (s, theta, zeta) x vpar and finds where |D| (and |iota*D|) is smallest.
     A near-zero D anywhere a real particle can reach is the physical cause of
     the runaway integration.

Run from the alpha_opt root, same as debug_trial16:
    python diagnose_trial16_field.py [path/to/config.yaml]

Outputs a summary to stdout and saves arrays to trial16_field_scan.npz
(in the working dir) for further offline inspection / plotting.

NOTE ON THE FIELD API: this script does not assume which derivative methods
exist. It probes the captured field object and uses whatever is available,
clearly reporting which quantities it could and could not obtain. If a needed
derivative (e.g. dKdtheta) is missing, the D reconstruction is skipped and the
script tells you so rather than guessing.
"""

import os
import sys
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Hardcoded x from the failing trial 16 (same as debug_trial16.py)
# ---------------------------------------------------------------------------
X = np.array([
    0.052528, 0.343611, 0.886642, 0.880989, 0.016367, 0.820174, 0.7586,   0.937056,
    0.050093, 0.448216, 0.671892, 0.156337, 0.394539, 0.484164, 0.486364, 0.672674,
    0.928225, 0.822357, 0.417637, 0.213327, 0.200875, 0.892966, 0.194755,
])


# ---------------------------------------------------------------------------
# Defensive field-quantity access
# ---------------------------------------------------------------------------
def _try_call(field, name):
    """Call field.<name>() and return a flat numpy array, or None if it fails."""
    fn = getattr(field, name, None)
    if fn is None:
        return None
    try:
        val = fn()
        return np.asarray(val).flatten()
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] field.{name}() raised {type(exc).__name__}: {exc}")
        return None


def gather_field_quantities(field, pts):
    """Set points on the field and pull every quantity we can.

    Returns a dict name -> flat array (only for quantities that succeeded).
    """
    field.set_points(pts)
    wanted = [
        "modB", "dmodBds", "dmodBdtheta", "dmodBdzeta",
        "G", "dGds",
        "I", "dIds",
        "iota", "diotads",
        "K", "dKdtheta", "dKdzeta",
    ]
    out = {}
    for name in wanted:
        arr = _try_call(field, name)
        if arr is not None:
            out[name] = arr
    return out


def summarize(name, arr):
    finite = np.isfinite(arr)
    nan = int(np.isnan(arr).sum())
    inf = int(np.isinf(arr).sum())
    if finite.any():
        amin = float(np.min(arr[finite]))
        amax = float(np.max(arr[finite]))
        aabsmin = float(np.min(np.abs(arr[finite])))
    else:
        amin = amax = aabsmin = float("nan")
    print(f"  {name:12s}: min={amin: .4e}  max={amax: .4e}  "
          f"|.|min={aabsmin: .4e}  NaN={nan}  Inf={inf}")
    return dict(min=amin, max=amax, absmin=aabsmin, nan=nan, inf=inf)


# ---------------------------------------------------------------------------
# Main diagnostic, called with the intercepted GPU-call arguments
# ---------------------------------------------------------------------------
def diagnose(field, kwargs, vmec):
    print("\n" + "=" * 70)
    print("TRIAL 16 FIELD / DENOMINATOR DIAGNOSTIC")
    print("=" * 70)

    srange = tuple(kwargs["srange"])
    trange = tuple(kwargs["trange"])
    zrange = tuple(kwargs["zrange"])
    stz_init = np.asarray(kwargs["stz_init"])
    vtang = np.asarray(kwargs["vtang"]).flatten()

    # The real physics constants, straight from the intercepted call:
    m = float(kwargs["m"])
    q = float(kwargs["q"])
    vtotal = float(kwargs["vtotal"])
    psi0 = float(kwargs["psi0"]) if "psi0" in kwargs else float(getattr(field, "psi0", np.nan))

    print(f"\nIntercepted tracing parameters:")
    print(f"  mass m      = {m: .6e}")
    print(f"  charge q    = {q: .6e}")
    print(f"  v_total     = {vtotal: .6e}")
    print(f"  psi0        = {psi0: .6e}")
    print(f"  nparticles  = {kwargs.get('nparticles')}")
    print(f"  srange = {srange}")
    print(f"  trange = {trange}")
    print(f"  zrange = {zrange}")

    # ---- 1. VMEC equilibrium quality --------------------------------------
    print("\n--- 1. VMEC equilibrium quality ---")
    try:
        w = vmec.wout
        for attr in ("aspect", "betatotal", "volavgB", "Rmajor_p", "Aminor_p"):
            if hasattr(w, attr):
                print(f"  {attr:12s}= {getattr(w, attr)}")
        if hasattr(w, "iotaf"):
            iotaf = np.asarray(w.iotaf).flatten()
            print(f"  iota profile (axis..edge): "
                  f"{iotaf[0]: .4f} ... {iotaf[-1]: .4f}  "
                  f"(min={iotaf.min(): .4f}, max={iotaf.max(): .4f})")
            if (iotaf.min() < 0) and (iotaf.max() > 0):
                print("  *** iota CHANGES SIGN in the profile -> "
                      "1/iota is singular somewhere. Strong suspect. ***")
        # Mercier stability if present
        for attr in ("DMerc", "DMerc_p"):
            if hasattr(w, attr):
                dm = np.asarray(getattr(w, attr)).flatten()
                print(f"  {attr}: min={np.nanmin(dm): .3e} max={np.nanmax(dm): .3e}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not read VMEC wout details: {exc}")

    # ---- 2. Initial-condition sanity --------------------------------------
    print("\n--- 2. Initial conditions (stz_init) ---")
    s0 = stz_init[:, 0]
    print(f"  s     : min={s0.min(): .6f}  max={s0.max(): .6f}")
    print(f"  vtang : min={vtang.min(): .4e}  max={vtang.max(): .4e}  "
          f"(|vtang|/v_total max = {np.max(np.abs(vtang))/vtotal: .4f})")
    print(f"  particles with s>=1 at init: {(s0 >= 1.0).sum()}")

    # ---- 3. Full-domain field scan ----------------------------------------
    print("\n--- 3. Field quantities over the FULL (s,theta,zeta) grid ---")
    ns = max(int(srange[2]), 40)
    nt = max(int(trange[2]), 40)
    nz = max(int(zrange[2]), 40)
    # avoid exact s=0 (axis) and s=1 (edge) where coords can be singular
    s_grid = np.linspace(max(srange[0], 1e-4), min(srange[1], 0.999), ns)
    t_grid = np.linspace(trange[0], trange[1], nt)
    z_grid = np.linspace(zrange[0], zrange[1], nz)
    S, T, Z = np.meshgrid(s_grid, t_grid, z_grid, indexing="ij")
    pts = np.stack([S.ravel(), T.ravel(), Z.ravel()], axis=1)
    print(f"  scanning {pts.shape[0]} grid points "
          f"(ns={ns}, nt={nt}, nz={nz})")

    fq = gather_field_quantities(field, pts)
    got = set(fq.keys())
    print(f"  obtained quantities: {sorted(got)}")
    missing = {"modB", "dmodBdtheta", "dmodBdzeta", "G", "dGds",
               "I", "dIds", "iota", "K", "dKdtheta", "dKdzeta"} - got
    if missing:
        print(f"  [note] could NOT obtain: {sorted(missing)}")

    print("\n  Field-quantity statistics over the grid:")
    stats = {}
    for name in sorted(fq):
        stats[name] = summarize(name, fq[name])

    # iota zero-crossing check over the grid
    if "iota" in fq:
        iota = fq["iota"]
        finite = np.isfinite(iota)
        if finite.any() and (iota[finite].min() < 0) and (iota[finite].max() > 0):
            print("  *** iota changes sign over the grid -> 1/iota singular. ***")
            near0 = np.nanargmin(np.abs(iota))
            print(f"      |iota| smallest at s={S.ravel()[near0]:.4f} "
                  f"theta={T.ravel()[near0]:.4f} zeta={Z.ravel()[near0]:.4f} "
                  f"iota={iota[near0]: .4e}")

    # ---- 4. Reconstruct D over (grid x vpar) ------------------------------
    print("\n--- 4. RHS denominator D over (grid x vpar) ---")
    needed = {"modB", "G", "I", "iota", "dGds", "dIds", "dKdtheta", "dKdzeta"}
    if not needed.issubset(got):
        print(f"  SKIPPED: need {sorted(needed)} but missing "
              f"{sorted(needed - got)}.")
        print("  -> tell me the field's derivative method names and I'll adapt.")
    else:
        modB = fq["modB"]
        G = fq["G"]
        I = fq["I"]
        iota = fq["iota"]
        # psi-derivatives = s-derivatives / psi0  (matches the kernel)
        dGdpsi = fq["dGds"] / psi0
        dIdpsi = fq["dIds"] / psi0
        dKdtheta = fq["dKdtheta"]
        dKdzeta = fq["dKdzeta"]

        vpar_grid = np.linspace(-vtotal, vtotal, 21)

        global_absmin_D = np.inf
        global_absmin_iotaD = np.inf
        worst = None
        worst_iotaD = None

        for vpar in vpar_grid:
            C = (-m * vpar * dKdzeta / modB
                 - q * iota
                 + m * vpar * dGdpsi / modB)
            F = (-m * vpar * dKdtheta / modB
                 + q
                 + m * vpar * dIdpsi / modB)
            D = (F * G - C * I) / iota
            iotaD = iota * D

            finiteD = np.isfinite(D)
            if finiteD.any():
                k = np.nanargmin(np.abs(np.where(finiteD, D, np.inf)))
                if np.abs(D[k]) < global_absmin_D:
                    global_absmin_D = float(np.abs(D[k]))
                    worst = (float(S.ravel()[k]), float(T.ravel()[k]),
                             float(Z.ravel()[k]), float(vpar),
                             float(D[k]), float(C[k]), float(F[k]),
                             float(G[k]), float(I[k]), float(iota[k]))
            finiteID = np.isfinite(iotaD)
            if finiteID.any():
                k2 = np.nanargmin(np.abs(np.where(finiteID, iotaD, np.inf)))
                if np.abs(iotaD[k2]) < global_absmin_iotaD:
                    global_absmin_iotaD = float(np.abs(iotaD[k2]))
                    worst_iotaD = (float(S.ravel()[k2]), float(T.ravel()[k2]),
                                   float(Z.ravel()[k2]), float(vpar),
                                   float(iotaD[k2]))

        print(f"  smallest |D|      over scan = {global_absmin_D: .4e}")
        if worst is not None:
            print(f"    at s={worst[0]:.4f} theta={worst[1]:.4f} "
                  f"zeta={worst[2]:.4f} vpar={worst[3]: .4e}")
            print(f"    D={worst[4]: .4e}  C={worst[5]: .4e}  F={worst[6]: .4e}  "
                  f"G={worst[7]: .4e}  I={worst[8]: .4e}  iota={worst[9]: .4e}")
        print(f"  smallest |iota*D| over scan = {global_absmin_iotaD: .4e}")
        if worst_iotaD is not None:
            print(f"    at s={worst_iotaD[0]:.4f} theta={worst_iotaD[1]:.4f} "
                  f"zeta={worst_iotaD[2]:.4f} vpar={worst_iotaD[3]: .4e}  "
                  f"iota*D={worst_iotaD[4]: .4e}")

        # Interpretation hint
        print("\n  Interpretation:")
        if global_absmin_iotaD < 1e-6 * (abs(q) * abs(np.nanmedian(G))):
            print("    A near-zero denominator EXISTS in the reachable domain.")
            print("    This is a genuine coordinate/orbit singularity for THIS")
            print("    configuration -> the integrator must reject/limit steps")
            print("    near it, and a particle wandering there is what blows up s.")
        else:
            print("    No obviously tiny denominator found on this grid sample.")
            print("    The runaway may occur between grid points, or be driven")
            print("    by a different term (e.g. modB near zero, or a stiff")
            print("    region). Consider a finer scan near the worst point above.")

        # save for offline plotting
        # try:
        #     np.savez(
        #         "trial16_field_scan.npz",
        #         s_grid=s_grid, t_grid=t_grid, z_grid=z_grid,
        #         modB=modB, G=G, I=I, iota=iota,
        #         dGdpsi=dGdpsi, dIdpsi=dIdpsi,
        #         dKdtheta=dKdtheta, dKdzeta=dKdzeta,
        #         m=m, q=q, vtotal=vtotal, psi0=psi0,
        #         vpar_grid=vpar_grid,
        #     )
        #     print("\n  Saved arrays to trial16_field_scan.npz")
        # except Exception as exc:  # noqa: BLE001
        #     print(f"  [warn] could not save npz: {exc}")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Harness (mirrors debug_trial16.py up to the GPU call, then diverts)
# ---------------------------------------------------------------------------
def make_diagnostic_patch(field_holder, vmec_holder):
    def patched(**kwargs):
        field = field_holder[0]
        vmec = vmec_holder[0]
        if field is None:
            print("[error] field was not captured; cannot run diagnostic.")
            sys.exit(1)
        diagnose(field, kwargs, vmec)
        # do NOT call the real GPU kernel
        sys.exit(0)
    return patched


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "alpha_opt/configs/garabedian_vmec_constraint.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for k in ["t_max", "tau", "tol", "max_B_target", "x_min"]:
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])

    aspect_ratio = cfg["aspect_ratio"]
    minor_radius = (
        float(cfg["minor_radius"])
        if cfg.get("minor_radius") is not None
        else 3.1 / aspect_ratio ** 0.38
    )
    major_radius = minor_radius * aspect_ratio

    from vmecpp.simsopt_compat import Vmec
    vmec = Vmec(cfg["vmec_input_file"], verbose=True)
    avg_B_estimate = cfg["max_B_target"] / np.sqrt(2)
    phiedge_high = np.pi * avg_B_estimate * minor_radius ** 2 * 2
    vmec.set("phiedge", phiedge_high)

    from alpha_opt.surface import SurfaceGarabedianQuantiles, SurfaceGarabedianLinear
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
        raise ValueError(f"Unsupported parameterization: {cfg['parameterization']}")

    import alpha_opt.tracing.alpha_tracing as _at

    field_holder = [None]
    vmec_holder = [vmec]
    _orig_gen = _at.generate_interpolant_and_initial_conditions

    def _capturing_gen(*args, **kwargs):
        result = _orig_gen(*args, **kwargs)
        field_holder[0] = result[-1]  # field is the last element of the tuple
        return result

    _at.generate_interpolant_and_initial_conditions = _capturing_gen

    _real_gpu = _at.firm3dpp.boozer_gpu_tracing
    _at.firm3dpp.boozer_gpu_tracing = make_diagnostic_patch(field_holder, vmec_holder)

    from alpha_opt.objective import get_objective
    x_scale = np.ones(len(surface.x))

    eval_dir = "diagnose_trial16_eval"
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
        vmec, surface, x_scale, raw_objective,
        max_B=cfg["max_B_target"],
        max_B_iterations=cfg["max_B_iterations"],
        phiedge=phiedge_high,
    )

    print(f"Running objective with hardcoded x (dim={len(X)})...")
    try:
        objective(X)
    except SystemExit:
        pass  # raised by the diagnostic patch after it finishes
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"Exception before/during diagnostic: {e}")
        traceback.print_exc()

    _at.firm3dpp.boozer_gpu_tracing = _real_gpu
    _at.generate_interpolant_and_initial_conditions = _orig_gen


if __name__ == "__main__":
    main()