import os
import numpy as np
import h5py
from scipy import optimize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, FunctionTransformer
from sklearn.decomposition import PCA
from simsopt.geo import SurfaceGarabedian, SurfaceRZFourier, Surface

pca_Garabedian_data_file = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "20251014-01-get_bounds_on_Garabedian_Deltas_PCA_inputs_5x5_withoutNfp123QIs.dat",
    )
)
pca_real_space_data_file = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "20251111-01_save_pca_shape_data_real_space_PCA_inputs.dat",
    )
)

weighted_pca_data_file = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "20260226-05_weighted_PCA_data.h5",
    )
)

class SurfacePCAGarabedian(Surface):
    """A Surface where the dofs are the amplitudes of principal components from
    a set of familiar stellarator shapes.

    The PCA is applied to Garabedian coefficients.

    You can set transform1=None to skip the first transformation.
    """
    def __init__(
            self, 
            nfp,
            major_radius,
            minor_radius,
            dimension,
            filename=pca_Garabedian_data_file,
            transform1=QuantileTransformer(),
            transform2=RobustScaler(),
        ):
        self._major_radius = major_radius
        self._minor_radius = minor_radius
        self.nfp = nfp
        self.dimension = dimension

        # Read the header line
        with open(filename, 'r') as f:
            header_line = f.readline().strip()
            # Split by whitespace to get column names
            column_names = header_line.split()
        # Drop the "#" at the start of the first column name if present
        if column_names[0].startswith("#"):
            column_names = column_names[1:]

        print(f"Column names: {column_names}")
        print(f"Number of columns: {len(column_names)}")

        # Load the data (skip header row)
        data = np.loadtxt(filename, skiprows=1)

        print(f"Data shape: {data.shape}")

        if transform1 is None:
            transform1 = FunctionTransformer()  # The identity transform

        # Set up a pipeline with Transformer followed by PCA
        self.pipeline = Pipeline([
            # ('transform', FunctionTransformer()),  # The identity transform
            # ('transform', RobustScaler()),
            ('transform', transform1),
            # ('transform', QuantileTransformer(output_distribution='normal')),
            ('pca', PCA()),
            ('transform2', transform2),
            # ('transform2', QuantileTransformer()),
        ])

        # Fit the pipeline to the data
        self.pipeline.fit(data)

        # Get the number of principal components
        self.n_components = self.pipeline.named_steps['pca'].n_components_
        print(f"Number of principal components: {self.n_components}")

        mpol_crop = 5
        ntor_crop = 5
        self.mmax = mpol_crop + 1
        self.mmin = -mpol_crop + 1
        self.nmax = ntor_crop
        self.nmin = -ntor_crop

        self.surface_garabedian = SurfaceGarabedian(
            nfp=self.nfp,
            mmax=self.mmax,
            nmax=self.nmax,
            mmin=self.mmin,
            nmin=self.nmin,
        )

        x0 = np.zeros(self.n_components)
        fixed = np.full(self.n_components, True)
        fixed[:self.dimension] = False  # Unfix the first 'dimension' components
        super().__init__(x0=x0, fixed=fixed)

    def recompute_bell(self, parent=None):
        # Transform back to original space
        PCA_space = self.pipeline.named_steps['transform2'].inverse_transform(self.local_full_x.reshape(1, -1))
        quantile_space = self.pipeline.named_steps['pca'].inverse_transform(PCA_space)
        original_space = self.pipeline.named_steps['transform'].inverse_transform(quantile_space)

        # Update the Garabedian surface
        j_col = 0
        for m in range(self.mmin, self.mmax + 1):
            for n in range(self.nmin, self.nmax + 1):
                if m == 0 and n == 0:
                    self.surface_garabedian.set_Delta(m, n, self._minor_radius)
                elif m == 1 and n == 0:
                    self.surface_garabedian.set_Delta(m, n, self._major_radius)
                else:
                    self.surface_garabedian.set_Delta(m, n, original_space[0, j_col] * self._minor_radius)
                    j_col += 1

        self.surface_rz_fourier = self.surface_garabedian.to_RZFourier()

    def to_RZFourier(self):
        return self.surface_rz_fourier

class SurfacePCARealSpace(Surface):
    """A Surface where the dofs are the amplitudes of principal components from
    a set of familiar stellarator shapes.

    The PCA is applied to (R, Z) values.

    You can set transform1=None to skip the first transformation.
    """
    def __init__(
            self, 
            nfp,
            major_radius,
            minor_radius,
            dimension,
            filename=pca_real_space_data_file,
            transform1=None,
            transform2=RobustScaler(),
            mpol=6,
            ntor=6,
        ):
        self._major_radius = major_radius
        self._minor_radius = minor_radius
        self.nfp = nfp
        self.dimension = dimension

        if isinstance(transform2, QuantileTransformer):
            # If transform2 gives results on [0, 1]:
            self.default_amplitude = 0.5
        else:
            # If transform2 gives results with mean 0:
            self.default_amplitude = 0.0

        print("Default amplitude for PCA surface:", self.default_amplitude)

        # Load the data (skip header row)
        data = np.loadtxt(filename)
        # n_theta and n_phi must match those used when generating the data
        self.n_theta = 32
        self.n_phi = 33

        print(f"Data shape: {data.shape}")
        assert data.shape[1] == self.n_theta * self.n_phi * 2

        if transform1 is None:
            transform1 = FunctionTransformer()  # The identity transform

        # Set up a pipeline with Transformer followed by PCA
        self.pipeline = Pipeline([
            # ('transform', FunctionTransformer()),  # The identity transform
            # ('transform', RobustScaler()),
            ('transform', transform1),
            # ('transform', QuantileTransformer(output_distribution='normal')),
            ('pca', PCA()),
            ('transform2', transform2),
            # ('transform2', QuantileTransformer()),
        ])

        # Fit the pipeline to the data
        self.pipeline.fit(data)

        # Get the number of principal components
        self.n_components = self.pipeline.named_steps['pca'].n_components_
        print(f"Number of principal components: {self.n_components}")

        self.surface = SurfaceRZFourier.from_nphi_ntheta(
            nfp=nfp,
            mpol=mpol,
            ntor=ntor,
            range="half period",
            ntheta=self.n_theta,
            nphi=self.n_phi,
        )

        x0 = np.zeros(self.n_components) + self.default_amplitude
        fixed = np.full(self.n_components, True)
        fixed[:self.dimension] = False  # Unfix the first 'dimension' components
        super().__init__(x0=x0, fixed=fixed)

    def recompute_bell(self, parent=None):
        # Transform back to original space
        PCA_space = self.pipeline.named_steps['transform2'].inverse_transform(self.local_full_x.reshape(1, -1))
        quantile_space = self.pipeline.named_steps['pca'].inverse_transform(PCA_space)
        original_space = self.pipeline.named_steps['transform'].inverse_transform(quantile_space)

        R = original_space[0, :self.n_theta * self.n_phi].reshape((self.n_phi, self.n_theta)) * self._minor_radius + self._major_radius
        Z = original_space[0, self.n_theta * self.n_phi:].reshape((self.n_phi, self.n_theta)) * self._minor_radius
        X = R * np.cos(2 * np.pi * self.surface.quadpoints_phi[:, None])
        Y = R * np.sin(2 * np.pi * self.surface.quadpoints_phi[:, None])
        gamma = np.concatenate((X[:, :, None], Y[:, :, None], Z[:, :, None]), axis=2)
        self.surface.least_squares_fit(gamma)

    def to_RZFourier(self):
        # Return a copy of the surface rather than the original surface so that
        # if e.g. change_resolution() is called, it doesn't break the internal state.
        return self.surface.copy()

class SurfaceWeightedPCA(Surface):
    """A Surface where the dofs are the amplitudes of principal components from
    a set of familiar stellarator shapes.

    The PCA is applied to (R, Z) values.

    You can set transform1=None to skip the first transformation.
    """
    def __init__(
            self, 
            nfp,
            major_radius,
            minor_radius,
            dimension,
            filename=weighted_pca_data_file,
            mpol=6,
            ntor=6,
            exact_radii=False,
        ):
        self._major_radius = major_radius
        self._minor_radius = minor_radius
        self.nfp = nfp
        self.dimension = dimension
        self.exact_radii = exact_radii

        with h5py.File(filename) as f:
            self.n_theta = f['n_theta'][()]
            self.n_phi = f['n_phi'][()]
            weights = f['weights'][()]
            data = f['data'][()]

        print(f"Data shape: {data.shape}")
        assert data.shape[1] == self.n_theta * self.n_phi * 2

        from weightedpca import WeightedPCA, WeightedQuantileTransformer

        # If more singular values than "dimension" are kept, their values will
        # not be set to 0, but rather they will be set to the median value of
        # the PCA coefficients in the dataset due to the quantile transformation.
        self.pca = WeightedPCA(dimension)
        # self.pca = WeightedPCA()
        pc_amplitudes = self.pca.fit_transform(data, sample_weight=weights)

        self.quantile_transformer = WeightedQuantileTransformer()
        self.quantile_transformer.fit(pc_amplitudes, sample_weight=weights)

        # Get the number of principal components
        self.n_components = self.pca.n_components_
        print(f"Number of principal components: {self.n_components}")

        self.surface = SurfaceRZFourier.from_nphi_ntheta(
            nfp=nfp,
            mpol=mpol,
            ntor=ntor,
            range="half period",
            ntheta=self.n_theta,
            nphi=self.n_phi,
        )

        x0 = np.full(self.n_components, 0.5)
        fixed = np.full(self.n_components, True)
        fixed[:self.dimension] = False  # Unfix the first 'dimension' components
        super().__init__(x0=x0, fixed=fixed)

    def recompute_bell(self, parent=None):
        # Transform back to original space
        PCA_space = self.quantile_transformer.inverse_transform(self.local_full_x.reshape(1, -1))
        print("PCA space:", PCA_space)
        original_space = self.pca.inverse_transform(PCA_space)

        R = original_space[0, :self.n_theta * self.n_phi].reshape((self.n_phi, self.n_theta)) * self._minor_radius + self._major_radius
        Z = original_space[0, self.n_theta * self.n_phi:].reshape((self.n_phi, self.n_theta)) * self._minor_radius
        X = R * np.cos(2 * np.pi * self.surface.quadpoints_phi[:, None])
        Y = R * np.sin(2 * np.pi * self.surface.quadpoints_phi[:, None])
        gamma = np.concatenate((X[:, :, None], Y[:, :, None], Z[:, :, None]), axis=2)
        self.surface.least_squares_fit(gamma)

        if self.exact_radii:
            # First vary Delta(1,0) to enforce exact aspect ratio. This is a single degree of freedom, so we can do a 1D root solve.
            target_aspect_ratio = self._major_radius / self._minor_radius

            def aspect_residual(x):
                x0 = self.surface.x.copy()
                x0[0] = x
                self.surface.x = x0
                return (
                    self.surface.aspect_ratio()
                    - target_aspect_ratio
                )

            try:
                root = optimize.newton(aspect_residual, x0=self._major_radius)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Failed to enforce exact radii with 1D root solve for Delta(1,0)."
                ) from exc

            aspect_residual(root)  # Set the major radius to the final value from the root solve

            # Now scale all the Fourier amplitudes to match the desired minor radius.
            scale = self._minor_radius / self.surface.minor_radius()
            self.surface.x = self.surface.x * scale
    def to_RZFourier(self):
        # Return a copy of the surface rather than the original surface so that
        # if e.g. change_resolution() is called, it doesn't break the internal state.
        return self.surface.copy()
