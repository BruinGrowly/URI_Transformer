"""
Shared Data Structures
"""

from dataclasses import dataclass, field
from typing import List, Dict
from src.phi_geometric_engine import PhiCoordinate
from enum import Enum


class QLAEDomain(Enum):
    ICE = "Consciousness"
    SFM = "Matter"
    IPE = "Life"
    PFE = "Energy"
    STM = "Information"
    PTD = "Space-Time"
    CCC = "Relationship"


@dataclass
class QLAEContext:
    """A structured context object from the QLAE framework."""
    domains: Dict[QLAEDomain, float]
    primary_domain: QLAEDomain
    is_valid: bool = True


class ExecutionStrategy(Enum):
    COMPASSIONATE_ACTION = "Compassionate Action"
    AUTHORITATIVE_COMMAND = "Authoritative Command"
    INSTRUCTIVE_GUIDANCE = "Instructive Guidance"
    CORRECTIVE_JUDGMENT = "Corrective Judgment"


@dataclass
class ExecutionPlan:
    """A structured plan for the Execution stage."""
    strategy: ExecutionStrategy
    magnitude: float
    description: str


@dataclass
class Intent:
    """Represents the Intent stage of the ICE framework."""
    purpose: str
    guiding_principles: List[str] = field(default_factory=list)


@dataclass
class Trajectory:
    """Represents the semantic trajectory between two coordinates."""
    velocity: PhiCoordinate
    acceleration: float


@dataclass
class TruthSenseResult:
    """The final, structured output of the TruthSenseTransformer."""
    raw_coord: PhiCoordinate
    aligned_coord: PhiCoordinate
    intent: Intent
    context: QLAEContext
    execution: ExecutionPlan
    final_output: str
    anchor_distance: float
    harmony_index: float
    semantic_integrity: float
    truth_sense_validation: bool
    deception_score: float
    foundational_principle: str
    trajectory: Trajectory
