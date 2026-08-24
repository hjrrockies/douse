# Do not delete this file. It tells python that douse is a module you can import from.
name = "douse"
__version__ = "0.4.0"

# modules available on package load
from . import polygon
from . import derivative
from . import objective
from . import evaluate
from . import optimize
from . import certify

# classes and functions available directly from douse
from .polygon import VertexPolygon, polygon_vertices, param_vector
from .derivative import (dlam_dp, dlam_dvertices, dlams_dp, cluster_matrices,
                         detect_cluster, directional_derivatives)
from .objective import (EigenvalueObjective, RatioObjective, SpectralMatch,
                        normalized_reciprocals, l2_loss)
from .evaluate import SpectrumEvaluator, ObjectiveFunction
from .optimize import descend, OptimizeResult
from .certify import first_order_report, second_order_probe, det_form
