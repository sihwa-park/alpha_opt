import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

from .loss_times import compute_energy_loss_fraction, time_at_which_energy_loss_exceeds
from .pca import PCASurface
from .surface import init_optimizable_surface
