import time
import numpy as np
from simsopt._core import ObjectiveFailure
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry, vmec_splines


def compute_max_B(vmec):
    """Get the maximum |B| on the boundary of a vmec equilibrium."""
    theta_grid = np.linspace(0, 2 * np.pi, 64)
    phi_grid = np.linspace(0, 2 * np.pi / vmec.boundary.nfp, 65)
    geom_data = vmec_compute_geometry(vmec_splines(vmec), 1, theta_grid, phi_grid)
    return np.max(geom_data.modB)

def get_objective(
    vmec,
    surface,
    x_scale,
    raw_objective,
    fail_val=1000.0,
    save_convergence_history=True,
    max_B=12.0,
    max_B_iterations=0,
    phiedge=None,
):
    """
    If max_B_iterations is 0, the field strength will be controlled by the
    original value of phiedge, and max_B is not used. If max_B_iterations is >0,
    vmec will be run that many additional times with phiedge varied each time to try to
    match the target max_B.

    if phiedge is None, the previous value of phiedge will be used for the first
    vmec calculation. Otherwise, phiedge will be set to the provided value before the
    first vmec calculation. Setting phiedge is useful when doing finite
    differences to ensure that the objective is independent of the history of
    past evaluations.
    """
    def objective(x):
        start_time = time.time()
        surface.x = x * x_scale
        surface2 = surface.to_RZFourier()
        # surface2.plot()

        with open("surface_parameters.txt", "a") as f_out:
            f_out.write(f"x = {[float(xj) for xj in x]}\n")
            f_out.write(f"x_scale = {[float(xj) for xj in x_scale]}\n")
            f_out.write(f"x_scaled = {[float(xj) for xj in surface.x]}\n")
            for name, value in zip(surface.local_dof_names, surface.x):
                f_out.write(f"{name:12}: {value}\n")
            f_out.write("\n")
            for name, value in zip(surface2.local_dof_names, surface2.x):
                f_out.write(f"{name:9}: {value}\n")

        if phiedge is not None:
            vmec.set('phiedge', phiedge)

        # This next line will be unnecessary once my PR to vmecpp is merged.
        surface2.change_resolution(vmec.indata.mpol, vmec.indata.ntor)
        vmec.boundary = surface2
        vmec.set_indata()
        indata_json = vmec.indata.model_dump_json(indent=2)
        with open("input.vmec_new", "w") as f:
            f.write(indata_json)
        # vmec.write_input("input.vmec")
        failure = False
        vmec.wout = None  # Clear out old wout data, if any.

        # original_stdout_fd = os.dup(sys.stdout.fileno())
        try:
            # Redirect stdout to a file
            # with open('output.txt', 'w') as output_file:
            #     os.dup2(output_file.fileno(), sys.stdout.fileno())
            #     # Run the VMEC simulation
            #     vmec.run()
            vmec.run()
            for _ in range(max_B_iterations):
                actual_max_B = compute_max_B(vmec)
                old_phiedge = vmec.get('phiedge')
                factor = max_B / actual_max_B
                new_phiedge = old_phiedge * factor
                print(f"Updating phiedge by a factor of {factor} from {old_phiedge} to {new_phiedge}")
                vmec.set('phiedge', new_phiedge)
                vmec.run()

        except ObjectiveFailure:
            # Some large value:
            failure = True

        if failure:
            f = fail_val
        else:
            f = raw_objective()

        if False:
            # Write force residual history to HDF5 file
            with h5py.File(f"force_residual_history.h5", "w") as hdf:
                hdf.create_dataset("force_residual_r", data=vmec.wout.force_residual_r)
                hdf.attrs["description"] = "Force residual (r component) vs iteration"
                hdf.create_dataset("force_residual_z", data=vmec.wout.force_residual_z)
                hdf.attrs["description"] = "Force residual (z component) vs iteration"
                hdf.create_dataset(
                    "force_residual_lambda", data=vmec.wout.force_residual_lambda
                )
                hdf.attrs[
                    "description"
                ] = "Force residual (lambda component) vs iteration"

        if save_convergence_history and vmec.wout is not None:
            # vmec.wout may not be changed from None if VMEC failed before iterating.
            with open("force_residual_history.txt", "w") as f_out:
                f_out.write(
                    "# Iteration, Force residual r, Force residual z, Force residual lambda\n"
                )
                for iter_num, (fr_r, fr_z, fr_lam) in enumerate(
                    zip(
                        vmec.wout.force_residual_r,
                        vmec.wout.force_residual_z,
                        vmec.wout.force_residual_lambda,
                    )
                ):
                    f_out.write(f"{iter_num:4d} {fr_r:.6e} {fr_z:.6e} {fr_lam:.6e}\n")

        with open("results.txt", "a") as f_out:
            f_out.write(f"x = {[float(xj) for xj in x]}\n")
            f_out.write(f"f = {f}\n")
            f_out.write(f"failed: {failure}\n")
            f_out.write(f"time: {time.time() - start_time}\n")

        return f

    return objective
