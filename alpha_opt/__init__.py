import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

from .pca import PCASurface
from .surface import init_optimizable_surface
