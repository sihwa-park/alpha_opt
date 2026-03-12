import os
import numpy as np
import h5py
from simsopt.geo import Surface, SurfaceRZFourier, SurfaceGarabedian
from weightedpca import WeightedQuantileTransformer

Garabedian_data_file = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "20260303-02_prepare_weighted_Garabedian_data_conservative_noNfp2.h5",
    )
)


"""Create an optimizable SurfaceGarabedian.

The aspect ratio, major radius, and minor radius are all fixed.
"""


def init_optimizable_surface(
    m_max,
    n_max,
    nfp,
    major_radius,
    minor_radius,
    elongation=2.0,
    scale=True,
    exponential_spectral_scaling_alpha=1.0,
    verbose=True,
):
    pre_surface = SurfaceRZFourier(
        mpol=m_max,
        ntor=n_max,
        nfp=nfp,
    )
    pre_surface.make_rotating_ellipse(
        major_radius=major_radius,
        minor_radius=minor_radius,
        elongation=elongation,
        torsion=minor_radius,
    )
    # pre_surface.change_resolution(mpol=m_max, ntor=n_max)

    # For Garabedian, don't set mmin = -mmax. Instead center them about 1.
    # surface = SurfaceGarabedian(mmax=2, mmin=0, nmax=1, nmin=-1)
    surface = SurfaceGarabedian.from_RZFourier(pre_surface)
    # surface.change_resolution(mmax=2, mmin=0, nmax=1, nmin=-1)
    if verbose:
        print("initial x:", surface.x)
        print(surface.local_dof_names)
    # exit(0)
    surface.set("Delta(1,0)", major_radius)  # Set major radius
    surface.set("Delta(0,0)", minor_radius)  # Set minor radius
    surface.fix("Delta(0,0)")  # Minor radius
    surface.fix("Delta(1,0)")  # Major radius
    dim_x = len(surface.x)
    if verbose:
        print("x:", surface.x)
        print("dof_names:", surface.dof_names)
    # vmec._should_save_outputs = True  # If you want wout files to be generated.

    # Compute x_scale for the dofs.
    if scale:
        # See ~/work24/20240415-01 Generating random stellarator boundary shapes.docx
        # exponential_spectral_scaling_alpha = 1.0
        ms = []
        ns = []
        for m in range(surface.mmin, surface.mmax + 1):
            for n in range(surface.nmin, surface.nmax + 1):
                if n == 0 and m in [0, 1]:
                    continue
                ms.append(m)
                ns.append(n)

        ms = np.array(ms)
        ns = np.array(ns)
        dof_names_should_be = [f"Delta({m},{n})" for m, n in zip(ms, ns)]
        assert (
            surface.local_dof_names == dof_names_should_be
        ), f"Expected {dof_names_should_be}, got {surface.local_dof_names}"
        x_scale = np.exp(
            -exponential_spectral_scaling_alpha * np.sqrt((ms - 1) ** 2 + ns**2)
        )
    else:
        x_scale = np.ones_like(surface.x)

    if verbose:
        print("x_scale:", x_scale)

    x0 = surface.x / x_scale

    return surface, dim_x, x_scale, x0


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
        filename=Garabedian_data_file,
    ):
        self._major_radius = major_radius
        self._minor_radius = minor_radius
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

        np.random.seed(0)
        self.transformer = WeightedQuantileTransformer()
        self.transformer.fit(data_variable, sample_weight=weights)

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

        self.surface_rz_fourier = self.surface_garabedian.to_RZFourier()

    def to_RZFourier(self):
        return self.surface_rz_fourier
