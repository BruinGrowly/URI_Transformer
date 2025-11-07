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
    """
    Explicit and dynamic implementation of the ICE framework with LJPW coupling awareness.

    Implements the layered ICE pipeline:
    - Intent (I): Focuses on Love + Wisdom
    - Context (C): Focuses on Justice, amplified by Love (κ_LJ = 1.4)
    - Execution (E): Focuses on Power, amplified by Love (κ_LP = 1.3)

    The coupling coefficients ensure that Love acts as a force multiplier,
    amplifying the effectiveness of Justice and Power.
    """
    def __init__(self, alignment_strength: float = 0.5, use_coupling: bool = True):
        """
        Initialize ICE Framework.

        Args:
            alignment_strength: How strongly to align to the principle (0.0 to 1.0)
            use_coupling: Whether to use coupling-aware alignment (default: True)
        """
        self.alignment_strength = alignment_strength
        self.use_coupling = use_coupling

    def process_intent(self, current_coord: PhiCoordinate, closest_principle: Principle) -> PhiCoordinate:
        """
        Process Intent layer: I = f(L+W)

        Intent focuses on Love and Wisdom. This establishes the core purpose.
        Love is the source dimension and is aligned directly without amplification.

        Args:
            current_coord: Current PhiCoordinate
            closest_principle: Closest matching principle

        Returns:
            PhiCoordinate with aligned Love and Wisdom
        """
        # Move current_coord's Love and Wisdom towards the principle's Love and Wisdom
        new_love = (1 - self.alignment_strength) * current_coord.love + self.alignment_strength * closest_principle.coordinate.love
        new_wisdom = (1 - self.alignment_strength) * current_coord.wisdom + self.alignment_strength * closest_principle.coordinate.wisdom

        # If coupling is enabled, consider Love's amplification of Wisdom (κ_LW = 1.5)
        if self.use_coupling:
            # The effective wisdom is amplified by love
            # We maintain the raw wisdom value but the effective impact is higher
            pass  # Wisdom coupling is applied in effective_dimensions calculations

        return PhiCoordinate(love=new_love, justice=current_coord.justice,
                             power=current_coord.power, wisdom=new_wisdom)

    def process_context(self, current_coord: PhiCoordinate, closest_principle: Principle) -> PhiCoordinate:
        """
        Process Context layer: C = f(J)

        Context focuses on Justice, evaluating fairness and moral alignment.
        When coupling is enabled, Justice is more effective when Love is present (κ_LJ = 1.4).

        Args:
            current_coord: Current PhiCoordinate
            closest_principle: Closest matching principle

        Returns:
            PhiCoordinate with aligned Justice
        """
        # Move current_coord's Justice towards the principle's Justice
        target_justice = closest_principle.coordinate.justice

        if self.use_coupling:
            # Justice is amplified by Love (κ_LJ = 1.4)
            # When aligning, consider the love-amplified target
            # Higher love makes justice alignment more effective
            love_amplification = 1 + 0.4 * current_coord.love
            # Scale the alignment strength by love amplification
            effective_strength = min(1.0, self.alignment_strength * love_amplification)
            new_justice = (1 - effective_strength) * current_coord.justice + effective_strength * target_justice
        else:
            new_justice = (1 - self.alignment_strength) * current_coord.justice + self.alignment_strength * target_justice

        return PhiCoordinate(love=current_coord.love, justice=new_justice,
                             power=current_coord.power, wisdom=current_coord.wisdom)

    def process_execution(self, current_coord: PhiCoordinate, closest_principle: Principle) -> PhiCoordinate:
        """
        Process Execution layer: E = f(P)

        Execution focuses on Power, determining the capacity to manifest intent.
        When coupling is enabled, Power is more effective when Love is present (κ_LP = 1.3).

        Args:
            current_coord: Current PhiCoordinate
            closest_principle: Closest matching principle

        Returns:
            PhiCoordinate with aligned Power
        """
        # Move current_coord's Power towards the principle's Power
        target_power = closest_principle.coordinate.power

        if self.use_coupling:
            # Power is amplified by Love (κ_LP = 1.3)
            # When aligning, consider the love-amplified target
            # Higher love makes power alignment more effective
            love_amplification = 1 + 0.3 * current_coord.love
            # Scale the alignment strength by love amplification
            effective_strength = min(1.0, self.alignment_strength * love_amplification)
            new_power = (1 - effective_strength) * current_coord.power + effective_strength * target_power
        else:
            new_power = (1 - self.alignment_strength) * current_coord.power + self.alignment_strength * target_power

        return PhiCoordinate(love=current_coord.love, justice=current_coord.justice,
                             power=new_power, wisdom=current_coord.wisdom)

    def get_effective_coordinate(self, coord: PhiCoordinate) -> dict:
        """
        Get the coupling-adjusted effective coordinate.

        This shows the actual effectiveness of each dimension when accounting
        for Love's force multiplier effect.

        Args:
            coord: PhiCoordinate to analyze

        Returns:
            Dictionary with effective dimensions
        """
        from src.ljpw_baselines import LJPWBaselines
        return LJPWBaselines.effective_dimensions(
            coord.love, coord.justice, coord.power, coord.wisdom
        )