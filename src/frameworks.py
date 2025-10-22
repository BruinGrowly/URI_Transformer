"""
Core Conceptual Frameworks (Evolved)
=====================================

Implements the evolved conceptual models for the TruthSense Transformer.
These frameworks now produce richer, structured outputs for a deeper
and more integrated ICE pipeline.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List
from src.phi_geometric_engine import PhiCoordinate

# --- QLAE Framework (Evolved) ---
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
    is_valid: bool = True # Will be moderated by the Justice axis

class QLAEFramework:
    """Evolved QLAE model for deep contextual analysis."""
    def get_context(self, coord: PhiCoordinate) -> QLAEContext:
        """Provides a structured, weighted contextual analysis."""
        scores = {
            QLAEDomain.ICE: (coord.love + coord.wisdom) / 2,
            QLAEDomain.SFM: (coord.power + coord.justice) / 2,
            QLAEDomain.IPE: coord.wisdom,
            QLAEDomain.PFE: coord.power,
            QLAEDomain.STM: (coord.wisdom + coord.justice + coord.love) / 3,
            QLAEDomain.PTD: coord.power,
            QLAEDomain.CCC: (coord.love + coord.justice) / 2,
        }
        # Sort domains by relevance
        sorted_domains = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        primary_domain = next(iter(sorted_domains))

        return QLAEContext(domains=sorted_domains, primary_domain=primary_domain)

# --- GOD Framework (Evolved) ---
class ExecutionStrategy(Enum):
    COMPASSIONATE_ACTION = "Compassionate Action"
    AUTHORITATIVE_COMMAND = "Authoritative Command"
    INSTRUCTIVE_GUIDANCE = "Instructive Guidance"
    CORRECTIVE_JUDGMENT = "Corrective Judgment"

@dataclass
class ExecutionPlan:
    """A structured plan for the Execution stage."""
    strategy: ExecutionStrategy
    magnitude: float # The power/feasibility of the plan
    description: str

class GODFramework:
    """Evolved GOD framework to generate a detailed ExecutionPlan."""
    def generate_plan(self, power_score: float, intent_coord: PhiCoordinate) -> ExecutionPlan:
        """Generates an ExecutionPlan based on the Power axis and intent."""
        # Determine strategy based on the original intent's dominant axis
        coords = {
            ExecutionStrategy.COMPASSIONATE_ACTION: intent_coord.love,
            ExecutionStrategy.AUTHORITATIVE_COMMAND: intent_coord.power,
            ExecutionStrategy.INSTRUCTIVE_GUIDANCE: intent_coord.wisdom,
            ExecutionStrategy.CORRECTIVE_JUDGMENT: intent_coord.justice,
        }
        strategy = max(coords, key=coords.get)

        description = f"Execute with {strategy.value}, leveraging a power capacity of {power_score:.2f}."

        return ExecutionPlan(
            strategy=strategy,
            magnitude=power_score,
            description=description
        )
