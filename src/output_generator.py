"""
Generative Output Module
========================

This module takes the structured outputs from the deep ICE pipeline
and synthesizes them into a new, meaningful, and context-aware
output sentence.
"""

from src.data_structures import Intent
from src.frameworks import QLAEContext, ExecutionPlan

class OutputGenerator:
    """Synthesizes structured ICE outputs into a final sentence."""

    def generate(self, intent: Intent, context: QLAEContext, execution: ExecutionPlan) -> str:
        """Generates a final output sentence."""

        # Start with the core intent
        intent_phrase = f"With {intent.purpose.lower()}"

        # Add context, moderated by Justice
        if not context.is_valid:
            context_phrase = "in a context of questionable truth"
        else:
            context_phrase = f"within the domain of '{context.primary_domain.value}'"

        # Add the execution plan
        execution_phrase = f"the recommended course of action is '{execution.strategy.value}' with a power magnitude of {execution.magnitude:.2f}."

        return f"{intent_phrase}, {context_phrase}, {execution_phrase}"
