import time
import numpy as np
from simsopt._core import ObjectiveFailure
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry, vmec_splines


class VmecConvergenceError(Exception):
    """Raised when VMEC fails to find a converged MHD equilibrium."""
    pass


def _compute_max_B(vmec):
    """Return the maximum |B| on the plasma boundary."""
    theta_grid = np.linspace(0, 2 * np.pi, 64)
    phi_grid = np.linspace(0, 2 * np.pi / vmec.boundary.nfp, 65)
    geom = vmec_compute_geometry(vmec_splines(vmec), 1, theta_grid, phi_grid)
    return np.max(geom.modB)

def get_objective(
    vmec,
    surface,
    x_scale,
    raw_objective,
    max_B=12.0,
    max_B_iterations=0,
    phiedge=None,
):
    """Build and return an objective callable.

    The returned function sets the surface DOFs, runs VMEC (optionally
    iterating to match a target max |B|), then calls ``raw_objective()``.

    Raises
    ------
    VmecConvergenceError
        If VMEC fails to converge for the given surface shape.  The caller
        is responsible for deciding how to handle this (e.g. skip the trial,
        assign a penalty, or record it separately).
    """

    def objective(x):
        t0 = time.time()

        # Update surface DOFs.
        surface.x = x * x_scale
        surface2 = surface.to_RZFourier()

        # Log surface parameters.
        with open("surface_parameters.txt", "a") as f:
            f.write(f"x = {[float(xi) for xi in x]}\n")
            f.write(f"x_scale = {[float(xj) for xj in x_scale]}\n")
            f.write(f"x_scaled = {[float(xj) for xj in surface.x]}\n")
            for name, val in zip(surface.local_dof_names, surface.x):
                f.write(f"  {name:12}: {val}\n")
            for name, val in zip(surface2.local_dof_names, surface2.x):
                f.write(f"  {name:9}: {val}\n")
            f.write("\n")

        if phiedge is not None:
            vmec.set("phiedge", phiedge)

        new_surf = surface2.change_resolution(vmec.indata.mpol, vmec.indata.ntor)
        if new_surf is not None:
            surface2 = new_surf

        vmec.boundary = surface2
        vmec.set_indata()
        with open("input.vmec_new", "w") as f:
            f.write(vmec.indata.model_dump_json(indent=2))

        vmec.wout = None  # clear stale wout

        # --- Run VMEC ---
        vmec_t0 = time.time()
        try:
            vmec.run()
            for _ in range(max_B_iterations):
                actual_max_B = _compute_max_B(vmec)
                factor = max_B / actual_max_B
                new_phiedge = vmec.get("phiedge") * factor
                print(
                    f"Updating phiedge by a factor of {factor} from "
                    f"{vmec.get('phiedge')} to {new_phiedge}"
                )
                vmec.set("phiedge", new_phiedge)
                vmec.run()
        except ObjectiveFailure as exc:
            print(f"Time to run vmec: {time.time() - vmec_t0:.3f}s (FAILED)")
            raise VmecConvergenceError("VMEC did not converge.") from exc

        print(f"Time to run vmec: {time.time() - vmec_t0:.3f}s")

        # --- Save convergence history ---
        if vmec.wout is not None:
            with open("force_residual_history.txt", "w") as f:
                f.write(
                    "# Iteration, Force residual r, Force residual z, "
                    "Force residual lambda\n"
                )
                for i, (fr_r, fr_z, fr_lam) in enumerate(
                    zip(
                        vmec.wout.force_residual_r,
                        vmec.wout.force_residual_z,
                        vmec.wout.force_residual_lambda,
                    )
                ):
                    f.write(f"{i:4d} {fr_r:.6e} {fr_z:.6e} {fr_lam:.6e}\n")

        # --- Evaluate tracing objective ---
        raw_t0 = time.time()
        result, case, obj_var = raw_objective()
        print(f"Time to call raw_objective: {time.time() - raw_t0:.3f}s")

        with open("results.txt", "a") as f:
            f.write(f"x = {[float(xi) for xi in x]}\n")
            f.write(f"f = {result}\n")
            f.write(f"time: {time.time() - t0:.3f}s\n\n")

        return result, case, obj_var

    return objective
