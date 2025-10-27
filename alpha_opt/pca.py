import os
import numpy as np
from sklearn import pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, FunctionTransformer
from sklearn.decomposition import PCA
from simsopt.geo import SurfaceGarabedian, Surface

pca_data_file = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "20251014-01-get_bounds_on_Garabedian_Deltas_PCA_inputs_5x5_withoutNfp123QIs.dat",
    )
)

class PCASurface(Surface):
    """A Surface where the dofs are the amplitudes of principal components from
    a set of familiar stellarator shapes.
    """
    def __init__(
            self, 
            nfp,
            major_radius,
            minor_radius,
            dimension,
            filename=pca_data_file,
            transform1=QuantileTransformer(),
            transform2=RobustScaler(),
        ):
        self.major_radius = major_radius
        self.minor_radius = minor_radius
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
                    self.surface_garabedian.set_Delta(m, n, self.minor_radius)
                elif m == 1 and n == 0:
                    self.surface_garabedian.set_Delta(m, n, self.major_radius)
                else:
                    self.surface_garabedian.set_Delta(m, n, original_space[0, j_col] * self.minor_radius)
                    j_col += 1

        self.surface_rz_fourier = self.surface_garabedian.to_RZFourier()

    def to_RZFourier(self):
        return self.surface_rz_fourier
