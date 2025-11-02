"""
Core Conceptual Frameworks
"""

from src.data_structures import (
    QLAEContext,
    QLAEDomain,
    ExecutionPlan,
    ExecutionStrategy
)
from src.phi_geometric_engine import PhiCoordinate

class QLAEFramework:
    """QLAE model for deep contextual analysis."""
    def get_context(self, coord: PhiCoordinate) -> QLAEContext:
        """Provides a structured, weighted contextual analysis."""
        scores = {
            QLAEDomain.ICE: (coord.love + coord.wisdom) / 2,
            QLAEDomain.SFM: (coord.power + coord.justice) / 2,
            QLAEDomain.IPE: coord.wisdom,
            QLAEDomain.PFE: coord.power,
            QLAEDomain.STM: (
                (coord.wisdom + coord.justice + coord.love) / 3
            ),
            QLAEDomain.PTD: coord.power,
            QLAEDomain.CCC: (coord.love + coord.justice) / 2,
        }
        sorted_domains = dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        primary_domain = next(iter(sorted_domains))

        return QLAEContext(
            domains=sorted_domains, primary_domain=primary_domain
        )

class GODFramework:
    """GOD framework to generate a detailed ExecutionPlan."""
    def generate_plan(
        self, power_score: float, intent_coord: PhiCoordinate
    ) -> ExecutionPlan:
        """Generates an ExecutionPlan based on the Power axis and intent."""
        coords = {
            ExecutionStrategy.COMPASSIONATE_ACTION: intent_coord.love,
            ExecutionStrategy.AUTHORITATIVE_COMMAND: intent_coord.power,
            ExecutionStrategy.INSTRUCTIVE_GUIDANCE: intent_coord.wisdom,
            ExecutionStrategy.CORRECTIVE_JUDGMENT: intent_coord.justice,
        }
        strategy = max(coords, key=coords.get)

        description = (
            f"Execute with {strategy.value}, leveraging a power "
            f"capacity of {power_score:.2f}."
        )

        return ExecutionPlan(
            strategy=strategy,
            magnitude=power_score,
            description=description
        )
