"""
Data Structures
===============

This module defines the core data structures used throughout the
TruthSenseTransformer pipeline.
"""

from dataclasses import dataclass
from typing import List
from src.phi_geometric_engine import PhiCoordinate
from src.frameworks import QLAEContext, ExecutionPlan


@dataclass
class Intent:
    """A structured object for the Intent stage."""
    purpose: str
    guiding_principles: List[str]


@dataclass
class TruthSenseResult:
    """A comprehensive result object for the deep ICE pipeline."""
    # Core Components
    raw_coord: PhiCoordinate
    aligned_coord: PhiCoordinate
    intent: Intent
    context: QLAEContext
    execution: ExecutionPlan

    # Final Output
    final_output: str

    # Metrics
    anchor_distance: float
    semantic_integrity: float
    truth_sense_validation: bool
