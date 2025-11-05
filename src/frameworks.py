"""
Core Conceptual Frameworks
"""

from src.data_structures import (
    QLAEContext,
    QLAEDomain,
    ExecutionPlan,
    ExecutionStrategy,
    Principle
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


class ICEFramework:
    """Explicit and dynamic implementation of the ICE framework."""
    def __init__(self, alignment_strength: float = 0.5):
        self.alignment_strength = alignment_strength # How strongly to align to the principle

    def process_intent(self, current_coord: PhiCoordinate, closest_principle: Principle) -> PhiCoordinate:
        # Intent focuses on Love and Wisdom (I = f(L+W))
        # Move current_coord's Love and Wisdom towards the principle's Love and Wisdom
        new_love = (1 - self.alignment_strength) * current_coord.love + self.alignment_strength * closest_principle.coordinate.love
        new_wisdom = (1 - self.alignment_strength) * current_coord.wisdom + self.alignment_strength * closest_principle.coordinate.wisdom
        return PhiCoordinate(love=new_love, justice=current_coord.justice,
                             power=current_coord.power, wisdom=new_wisdom)

    def process_context(self, current_coord: PhiCoordinate, closest_principle: Principle) -> PhiCoordinate:
        # Context focuses on Justice (C = f(J))
        # Move current_coord's Justice towards the principle's Justice
        new_justice = (1 - self.alignment_strength) * current_coord.justice + self.alignment_strength * closest_principle.coordinate.justice
        return PhiCoordinate(love=current_coord.love, justice=new_justice,
                             power=current_coord.power, wisdom=current_coord.wisdom)

    def process_execution(self, current_coord: PhiCoordinate, closest_principle: Principle) -> PhiCoordinate:
        # Execution focuses on Power (E = f(P))
        # Move current_coord's Power towards the principle's Power
        new_power = (1 - self.alignment_strength) * current_coord.power + self.alignment_strength * closest_principle.coordinate.power
        return PhiCoordinate(love=current_coord.love, justice=current_coord.justice,
                             power=new_power, wisdom=current_coord.wisdom)