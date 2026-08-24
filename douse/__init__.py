# Do not delete this file. It tells python that douse is a module you can import from.
name = "douse"
__version__ = "0.3.0"

# modules available on package load
from . import polygon
from . import derivative
from . import objective
from . import evaluate
from . import optimize

# classes and functions available directly from douse
from .polygon import VertexPolygon, polygon_vertices, param_vector
from .derivative import dlam_dp, dlam_dvertices, dlams_dp
from .objective import EigenvalueObjective, SpectralMatch, normalized_reciprocals, l2_loss
from .evaluate import SpectrumEvaluator, ObjectiveFunction
from .optimize import descend, OptimizeResult
