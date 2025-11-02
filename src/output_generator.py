"""
Sophisticated Output Generator
"""

from src.data_structures import TruthSenseResult

class OutputGenerator:
    """A sophisticated output generator that creates a narrative report."""

    def synthesize_output(self, result: TruthSenseResult) -> str:
        """Generates a final, narrative output string."""

        narrative = []

        # 1. Executive Summary
        narrative.append(
            f"The input's core intent, '{result.intent.purpose}', has been analyzed within the "
            f"context of the '{result.context.primary_domain.value}' domain. The recommended "
            f"course of action is '{result.execution.strategy.value}'."
        )

        # 2. Foundational Principle
        narrative.append(
            f"\nThis analysis is grounded in the foundational principle of "
            f"'{result.foundational_principle}'. The system's alignment with this principle "
            f"is strong, with a harmony index of {result.harmony_index:.2f}."
        )

        # 3. Semantic Trajectory
        narrative.append(
            f"\nThe semantic trajectory from the raw input to the aligned concept shows a "
            f"significant shift, with an acceleration of {result.trajectory.acceleration:.2f}. "
            f"This indicates a successful alignment with the system's core principles."
        )

        # 4. Deception/Integrity Score
        if result.deception_score > 0.5:
            narrative.append(
                f"\nWARNING: A high deception score of {result.deception_score:.2f} was detected. "
                f"This suggests a potential misalignment between the stated intent and the "
                f"underlying meaning."
            )
        else:
            narrative.append(
                f"\nThe analysis confirms a high degree of semantic integrity, with a deception "
                f"score of {result.deception_score:.2f}."
            )

        return "\n".join(narrative)
