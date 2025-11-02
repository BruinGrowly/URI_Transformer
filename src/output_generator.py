"""
Simple Output Generator
"""

from src.data_structures import Intent, QLAEContext, ExecutionPlan

class OutputGenerator:
    """A simple output generator."""
    def generate(
        self,
        intent: Intent,
        context: QLAEContext,
        execution: ExecutionPlan
    ) -> str:
        """Generates a final output string."""
        return (
            f"With {intent.purpose}, within the domain of "
            f"'{context.primary_domain.value}', the recommended course of "
            f"action is '{execution.strategy.value}' with a power "
            f"magnitude of {execution.magnitude:.2f}."
        )
