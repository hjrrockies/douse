# Do not delete this file. It tells python that douse is a module you can import from.
name = "douse"
__version__ = "0.1.0"

# modules available on package load
from . import polygon
from . import derivative

# classes and functions available directly from douse
from .polygon import VertexPolygon, polygon_vertices, param_vector
from .derivative import dlam_dp, dlam_dvertices
