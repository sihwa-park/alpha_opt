import os
import numpy as np
import h5py
from scipy import optimize
from simsopt.geo import Surface, SurfaceRZFourier, SurfaceGarabedian
from weightedpca import WeightedQuantileTransformer

"""Create an optimizable SurfaceGarabedian.

The aspect ratio, major radius, and minor radius are all fixed.
"""
class SurfaceGarabedianQuantiles(Surface):
    """Similar to SurfaceGarabedian, but the dofs are scaled using a data
    distribution to lie in [0, 1].

    """

    def __init__(
        self,
        nfp,
        major_radius,
        minor_radius,
        mpol,
        ntor,
        filename=None,
        exact_radii=False,
        seed=0,
    ):
        self._major_radius = major_radius
        self._minor_radius = minor_radius
        self.exact_radii = exact_radii
        self.nfp = nfp
        self.mpol = mpol
        self.ntor = ntor

        self.mmax = mpol + 1
        self.mmin = -mpol + 1
        self.nmax = ntor
        self.nmin = -ntor

        self.surface_garabedian = SurfaceGarabedian(
            nfp=self.nfp,
            mmax=self.mmax,
            nmax=self.nmax,
            mmin=self.mmin,
            nmin=self.nmin,
        )

        with h5py.File(filename, "r") as f:
            data_all = f["data"][()]
            weights = f["weights"][()]
            ms_all = f["ms"][()]
            ns_all = f["ns"][()]

        print("Full data shape:", data_all.shape)
        print("Number of configurations:", data_all.shape[0])
        print("Number of Garabedian modes in file:", data_all.shape[1])

        mask_keep = (
            (ms_all >= -mpol + 1)
            & (ms_all <= mpol + 1)
            & (ns_all >= -ntor)
            & (ns_all <= ntor)
        )

        if not np.any(mask_keep):
            raise RuntimeError("No modes selected. Check mpol and ntor.")

        ms_selected = ms_all[mask_keep]
        ns_selected = ns_all[mask_keep]
        data_selected = data_all[:, mask_keep]

        # Sort selected columns by (m, n) for predictable slider ordering.
        sort_idx = np.lexsort((ns_selected, ms_selected))
        ms_selected = ms_selected[sort_idx]
        ns_selected = ns_selected[sort_idx]
        data_selected = data_selected[:, sort_idx]
        self.ms_selected = ms_selected
        self.ns_selected = ns_selected

        # Validate fixed modes are present in selected range.
        idx_00 = np.where((ms_selected == 0) & (ns_selected == 0))[0]
        idx_10 = np.where((ms_selected == 1) & (ns_selected == 0))[0]
        if idx_00.size != 1:
            raise RuntimeError(
                "Expected exactly one (m,n)=(0,0) mode in selected range."
            )
        if idx_10.size != 1:
            raise RuntimeError(
                "Expected exactly one (m,n)=(1,0) mode in selected range."
            )
        idx_00 = idx_00[0]
        idx_10 = idx_10[0]
        self.idx_00 = idx_00
        self.idx_10 = idx_10

        fixed_mask = ((ms_selected == 0) & (ns_selected == 0)) | (
            (ms_selected == 1) & (ns_selected == 0)
        )
        variable_mask = ~fixed_mask
        self.variable_mask = variable_mask

        self.ms_variable = ms_selected[variable_mask]
        self.ns_variable = ns_selected[variable_mask]
        data_variable = data_selected[:, variable_mask]

        print("Selected total modes:", data_selected.shape[1])
        print("Variable modes (number of dofs):", data_variable.shape[1])

        if data_variable.shape[1] == 0:
            raise RuntimeError(
                "No variable modes available after fixing (0,0) and (1,0)."
            )

        # Confirm that selected set is exactly the rectangular range required by this surface.
        expected_mode_set = {
            (m, n)
            for m in range(self.mmin, self.mmax + 1)
            for n in range(self.nmin, self.nmax + 1)
        }
        selected_mode_set = {(int(m), int(n)) for m, n in zip(ms_selected, ns_selected)}
        if selected_mode_set != expected_mode_set:
            missing = sorted(expected_mode_set - selected_mode_set)
            extra = sorted(selected_mode_set - expected_mode_set)
            raise RuntimeError(
                "Selected modes do not match required rectangular grid. "
                f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}, "
                f"Extra: {extra[:5]}{'...' if len(extra) > 5 else ''}"
            )

        ##########################################################
        # Transform variable Garabedian amplitudes only
        ##########################################################

        np.random.seed(seed)
        self.transformer = WeightedQuantileTransformer()
        self.transformer.fit(data_variable, weights=weights)

        x0 = np.full(len(self.ms_variable), 0.5)
        super().__init__(x0=x0)

    def recompute_bell(self, parent=None):
        transformed_vals = self.x.reshape(1, -1)
        variable_original = (
            self.transformer.inverse_transform(transformed_vals)[0, :]
            * self._minor_radius
        )

        full_values = np.zeros(self.ms_selected.size)
        full_values[self.idx_00] = self._minor_radius
        full_values[self.idx_10] = self._major_radius
        full_values[self.variable_mask] = variable_original

        mode_map = {
            (int(m), int(n)): float(v)
            for m, n, v in zip(self.ms_selected, self.ns_selected, full_values)
        }

        # Update the Garabedian surface
        for m in range(self.mmin, self.mmax + 1):
            for n in range(self.nmin, self.nmax + 1):
                self.surface_garabedian.set_Delta(m, n, mode_map[(m, n)])

        if self.exact_radii:
            # First vary Delta(1,0) to enforce exact aspect ratio. This is a single degree of freedom, so we can do a 1D root solve.
            target_aspect_ratio = self._major_radius / self._minor_radius

            def aspect_residual(x):
                self.surface_garabedian.set_Delta(1, 0, x)
                return (
                    self.surface_garabedian.to_RZFourier().aspect_ratio()
                    - target_aspect_ratio
                )

            try:
                root = optimize.newton(aspect_residual, x0=self._major_radius)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Failed to enforce exact radii with 1D root solve for Delta(1,0)."
                ) from exc

            self.surface_garabedian.set_Delta(1, 0, root)

            # Now scale all the Delta(m,n) parameters to match the desired minor radius.
            scale = self._minor_radius / self.surface_garabedian.to_RZFourier().minor_radius()
            self.surface_garabedian.x = self.surface_garabedian.x * scale

        self.surface_rz_fourier = self.surface_garabedian.to_RZFourier()

    def to_RZFourier(self):
        return self.surface_rz_fourier
    
class SurfaceGarabedianLinear(Surface):
    """Like SurfaceGarabedianQuantiles, but uses a linear [0,1] normalization.

    The [0,1] DOF space is mapped linearly to the [q05, q95] weighted quantile
    range of each Garabedian mode in the training data, so x=0 corresponds to
    the 5th percentile and x=1 to the 95th percentile.
    """

    def __init__(
        self,
        nfp,
        major_radius,
        minor_radius,
        mpol,
        ntor,
        filename=None,
        exact_radii=False,
        seed=0,
    ):
        self._major_radius = major_radius
        self._minor_radius = minor_radius
        self.exact_radii = exact_radii
        self.nfp = nfp
        self.mpol = mpol
        self.ntor = ntor

        self.mmax = mpol + 1
        self.mmin = -mpol + 1
        self.nmax = ntor
        self.nmin = -ntor

        self.surface_garabedian = SurfaceGarabedian(
            nfp=self.nfp,
            mmax=self.mmax,
            nmax=self.nmax,
            mmin=self.mmin,
            nmin=self.nmin,
        )

        with h5py.File(filename, "r") as f:
            data_all = f["data"][()]
            weights = f["weights"][()]
            ms_all = f["ms"][()]
            ns_all = f["ns"][()]

        print("Full data shape:", data_all.shape)
        print("Number of configurations:", data_all.shape[0])
        print("Number of Garabedian modes in file:", data_all.shape[1])

        mask_keep = (
            (ms_all >= -mpol + 1)
            & (ms_all <= mpol + 1)
            & (ns_all >= -ntor)
            & (ns_all <= ntor)
        )

        if not np.any(mask_keep):
            raise RuntimeError("No modes selected. Check mpol and ntor.")

        ms_selected = ms_all[mask_keep]
        ns_selected = ns_all[mask_keep]
        data_selected = data_all[:, mask_keep]

        sort_idx = np.lexsort((ns_selected, ms_selected))
        ms_selected = ms_selected[sort_idx]
        ns_selected = ns_selected[sort_idx]
        data_selected = data_selected[:, sort_idx]
        self.ms_selected = ms_selected
        self.ns_selected = ns_selected

        idx_00 = np.where((ms_selected == 0) & (ns_selected == 0))[0]
        idx_10 = np.where((ms_selected == 1) & (ns_selected == 0))[0]
        if idx_00.size != 1:
            raise RuntimeError(
                "Expected exactly one (m,n)=(0,0) mode in selected range."
            )
        if idx_10.size != 1:
            raise RuntimeError(
                "Expected exactly one (m,n)=(1,0) mode in selected range."
            )
        idx_00 = idx_00[0]
        idx_10 = idx_10[0]
        self.idx_00 = idx_00
        self.idx_10 = idx_10

        fixed_mask = ((ms_selected == 0) & (ns_selected == 0)) | (
            (ms_selected == 1) & (ns_selected == 0)
        )
        variable_mask = ~fixed_mask
        self.variable_mask = variable_mask

        self.ms_variable = ms_selected[variable_mask]
        self.ns_variable = ns_selected[variable_mask]
        data_variable = data_selected[:, variable_mask]

        print("Selected total modes:", data_selected.shape[1])
        print("Variable modes (number of dofs):", data_variable.shape[1])

        if data_variable.shape[1] == 0:
            raise RuntimeError(
                "No variable modes available after fixing (0,0) and (1,0)."
            )

        expected_mode_set = {
            (m, n)
            for m in range(self.mmin, self.mmax + 1)
            for n in range(self.nmin, self.nmax + 1)
        }
        selected_mode_set = {(int(m), int(n)) for m, n in zip(ms_selected, ns_selected)}
        if selected_mode_set != expected_mode_set:
            missing = sorted(expected_mode_set - selected_mode_set)
            extra = sorted(selected_mode_set - expected_mode_set)
            raise RuntimeError(
                "Selected modes do not match required rectangular grid. "
                f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}, "
                f"Extra: {extra[:5]}{'...' if len(extra) > 5 else ''}"
            )

        np.random.seed(seed)
        n_dofs = data_variable.shape[1]
        # w = weights /weights.sum(axis=0, keepdims=True)
        # sorted_idx = np.argsort(data_variable, axis=0)          # [n, d]
        # sorted_data = np.take_along_axis(data_variable, sorted_idx, axis=0)
        # cumw = np.cumsum(np.take_along_axis(w, sorted_idx, axis=0), axis=0)                # [n, d]
        # self.lower = np.array([np.interp(0.05, cumw[:, j], sorted_data[:, j]) for j in range(n_dofs)])
        # self.upper = np.array([np.interp(0.95, cumw[:, j], sorted_data[:, j]) for j in range(n_dofs)])
        self.lower = np.percentile(data_variable, 5, axis=0)
        self.upper = np.percentile(data_variable, 95, axis=0)

        x0 = np.full(n_dofs, 0.5)
        super().__init__(x0=x0)

    def recompute_bell(self, parent=None):
        variable_original = (
            self.lower + self.x * (self.upper - self.lower)
        ) * self._minor_radius

        full_values = np.zeros(self.ms_selected.size)
        full_values[self.idx_00] = self._minor_radius
        full_values[self.idx_10] = self._major_radius
        full_values[self.variable_mask] = variable_original

        mode_map = {
            (int(m), int(n)): float(v)
            for m, n, v in zip(self.ms_selected, self.ns_selected, full_values)
        }

        for m in range(self.mmin, self.mmax + 1):
            for n in range(self.nmin, self.nmax + 1):
                self.surface_garabedian.set_Delta(m, n, mode_map[(m, n)])

        if self.exact_radii:
            target_aspect_ratio = self._major_radius / self._minor_radius

            def aspect_residual(x):
                self.surface_garabedian.set_Delta(1, 0, x)
                return (
                    self.surface_garabedian.to_RZFourier().aspect_ratio()
                    - target_aspect_ratio
                )

            try:
                root = optimize.newton(aspect_residual, x0=self._major_radius)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Failed to enforce exact radii with 1D root solve for Delta(1,0)."
                ) from exc

            self.surface_garabedian.set_Delta(1, 0, root)

            scale = self._minor_radius / self.surface_garabedian.to_RZFourier().minor_radius()
            self.surface_garabedian.x = self.surface_garabedian.x * scale

        self.surface_rz_fourier = self.surface_garabedian.to_RZFourier()

    def to_RZFourier(self):
        return self.surface_rz_fourier