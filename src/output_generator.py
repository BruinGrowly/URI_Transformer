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
            "--- Executive Summary ---\n"
            f"The input's core intent, '{result.intent.purpose}', has been "
            f"analyzed within the context of '{result.context.primary_domain.value}'. "
            f"The recommended course of action, based on the principle of "
            f"'{result.foundational_principle.name}', is "
            f"'{result.execution.strategy.value}'."
        )

        # 2. Foundational Principle
        narrative.append(
            "\n--- Foundational Principle ---\n"
            f"This analysis is grounded in the principle of "
            f"'{result.foundational_principle.name}', which is defined as: "
            f"'{result.foundational_principle.description}'.\n"
            f"The system's alignment with this principle is strong, with a "
            f"harmony index of {result.harmony_index:.2f}."
        )

        # 3. Semantic Trajectory
        narrative.append(
            "\n--- Semantic Trajectory ---\n"
            f"The semantic trajectory from the raw input to the aligned "
            f"concept shows a significant shift, with an acceleration of "
            f"{result.trajectory.acceleration:.2f}. This indicates a "
            f"successful alignment with the system's core principles."
        )

        # 4. Deception/Integrity Score
        narrative.append("\n--- Deception/Integrity Score ---")
        if result.deception_score > 0.5:
            narrative.append(
                f"WARNING: A high deception score of "
                f"{result.deception_score:.2f} was detected. This suggests a "
                f"potential misalignment between the stated intent and the "
                f"underlying meaning, as the 'Justice' value of the input "
                f"({result.raw_coord.justice:.2f}) is low."
            )
        else:
            narrative.append(
                f"The analysis confirms a high degree of semantic integrity, "
                f"with a deception score of {result.deception_score:.2f}. "
                f"This is consistent with the input's high 'Justice' value "
                f"of {result.raw_coord.justice:.2f}."
            )

        return "\n".join(narrative)
