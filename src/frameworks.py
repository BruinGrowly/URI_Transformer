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

        summary = (
            f"Execute with {strategy.value}, leveraging a power "
            f"capacity of {power_score:.2f}."
        )

        steps = []
        outcome = ""
        principles = []

        if strategy == ExecutionStrategy.COMPASSIONATE_ACTION:
            steps = [
                "Identify the immediate needs of all involved.",
                "Provide resources and support to alleviate suffering.",
                "Foster an environment of empathy and understanding."
            ]
            outcome = "To restore emotional well-being and strengthen relationships."
            principles = ["Empathy", "Kindness", "Generosity"]
        elif strategy == ExecutionStrategy.AUTHORITATIVE_COMMAND:
            steps = [
                "Clearly define the objective and desired outcome.",
                "Issue clear, concise directives to all parties.",
                "Monitor progress and enforce compliance."
            ]
            outcome = "To establish order and achieve the objective efficiently."
            principles = ["Clarity", "Decisiveness", "Accountability"]
        elif strategy == ExecutionStrategy.INSTRUCTIVE_GUIDANCE:
            steps = [
                "Assess the knowledge gaps of the individuals involved.",
                "Provide clear, step-by-step instructions.",
                "Offer mentorship and opportunities for growth."
            ]
            outcome = "To empower individuals with new knowledge and skills."
            principles = ["Patience", "Clarity", "Mentorship"]
        elif strategy == ExecutionStrategy.CORRECTIVE_JUDGMENT:
            steps = [
                "Identify the specific violation of principles or rules.",
                "Administer a fair and proportionate consequence.",
                "Provide a path for remediation and restoration."
            ]
            outcome = "To uphold justice, correct wrongdoing, and restore balance."
            principles = ["Fairness", "Accountability", "Restoration"]

        return ExecutionPlan(
            strategy=strategy,
            magnitude=power_score,
            summary=summary,
            steps=steps,
            outcome=outcome,
            principles_to_uphold=principles
        )
