"""
URI-Transformer Main Module
===========================

This module initializes the URI-Transformer package and exposes
the core components of the new, refactored architecture.
"""

from .truth_sense_transformer import TruthSenseTransformer
from .data_structures import TruthSenseResult, Intent
from .frameworks import QLAEContext, ExecutionPlan
from .phi_geometric_engine import PhiCoordinate

__all__ = [
    "TruthSenseTransformer",
    "TruthSenseResult",
    "Intent",
    "QLAEContext",
    "ExecutionPlan",
    "PhiCoordinate",
]
