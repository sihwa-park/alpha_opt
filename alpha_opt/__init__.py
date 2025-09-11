import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

from .surface import init_optimizable_surface

def foo(x):
    return x + 1
