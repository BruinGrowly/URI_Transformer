"""
Unit tests for the OutputGenerator.
"""
import unittest
from src.output_generator import OutputGenerator
from src.data_structures import (
    TruthSenseResult,
    Intent,
    QLAEContext,
    ExecutionPlan,
    Trajectory,
    QLAEDomain,
    ExecutionStrategy,
)
from src.phi_geometric_engine import PhiCoordinate
from src.knowledge_graph import Principle


class TestOutputGenerator(unittest.TestCase):
    """Test suite for the OutputGenerator."""

    def test_synthesize_output_high_deception(self):
        """
        Tests that the output generator produces the correct narrative
        for a result with a high deception score.
        """
        result = TruthSenseResult(
            raw_coord=PhiCoordinate(0.2, 0.1, 0.8, 0.3),
            aligned_coord=PhiCoordinate(0.61, 0.55, 1.2, 0.8),
            intent=Intent(purpose="To act with benevolent purpose (Love: 0.61)"),
            context=QLAEContext(
                domains={QLAEDomain.PFE: 1.2},
                primary_domain=QLAEDomain.PFE
            ),
            execution=ExecutionPlan(
                strategy=ExecutionStrategy.AUTHORITATIVE_COMMAND,
                magnitude=1.2,
                description="Execute with Authoritative Command, leveraging a power capacity of 1.20."
            ),
            final_output="",
            anchor_distance=1.2,
            harmony_index=0.45,
            semantic_integrity=1.0,
            truth_sense_validation=False,
            deception_score=0.91,
            foundational_principle=Principle(
                name="Effective Power",
                description="The principle of capability, strength, and authority.",
                coordinate=PhiCoordinate(1.0, 1.2, 1.8, 1.4)
            ),
            trajectory=Trajectory(
                velocity=PhiCoordinate(0.41, 0.45, 0.4, 0.5),
                acceleration=0.55
            )
        )

        generator = OutputGenerator()
        output = generator.synthesize_output(result)

        self.assertIn("--- Executive Summary ---", output)
        self.assertIn("recommended course of action, based on the principle of 'Effective Power', is 'Authoritative Command'", output)
        self.assertIn("--- Foundational Principle ---", output)
        self.assertIn("grounded in the principle of 'Effective Power'", output)
        self.assertIn("The principle of capability, strength, and authority.", output)
        self.assertIn("harmony index of 0.45", output)
        self.assertIn("--- Semantic Trajectory ---", output)
        self.assertIn("acceleration of 0.55", output)
        self.assertIn("--- Deception/Integrity Score ---", output)
        self.assertIn("WARNING: A high deception score of 0.91 was detected.", output)
        self.assertIn("as the 'Justice' value of the input (0.10) is low.", output)

    def test_synthesize_output_low_deception(self):
        """
        Tests that the output generator produces the correct narrative
        for a result with a low deception score.
        """
        result = TruthSenseResult(
            raw_coord=PhiCoordinate(0.7, 0.9, 0.6, 0.8),
            aligned_coord=PhiCoordinate(1.13, 1.2, 1.0, 1.2),
            intent=Intent(purpose="To act with benevolent purpose (Love: 1.13)"),
            context=QLAEContext(
                domains={QLAEDomain.IPE: 1.2},
                primary_domain=QLAEDomain.IPE
            ),
            execution=ExecutionPlan(
                strategy=ExecutionStrategy.INSTRUCTIVE_GUIDANCE,
                magnitude=1.0,
                description="Execute with Instructive Guidance, leveraging a power capacity of 1.00."
            ),
            final_output="",
            anchor_distance=1.1,
            harmony_index=0.46,
            semantic_integrity=1.0,
            truth_sense_validation=True,
            deception_score=0.0,
            foundational_principle=Principle(
                name="Discerning Wisdom",
                description="The principle of knowledge, understanding, and insight.",
                coordinate=PhiCoordinate(1.6, 1.6, 1.4, 1.8)
            ),
            trajectory=Trajectory(
                velocity=PhiCoordinate(0.43, 0.3, 0.4, 0.4),
                acceleration=0.54
            )
        )

        generator = OutputGenerator()
        output = generator.synthesize_output(result)

        self.assertIn("--- Executive Summary ---", output)
        self.assertIn("recommended course of action, based on the principle of 'Discerning Wisdom', is 'Instructive Guidance'", output)
        self.assertIn("--- Foundational Principle ---", output)
        self.assertIn("grounded in the principle of 'Discerning Wisdom'", output)
        self.assertIn("The principle of knowledge, understanding, and insight.", output)
        self.assertIn("harmony index of 0.46", output)
        self.assertIn("--- Semantic Trajectory ---", output)
        self.assertIn("acceleration of 0.54", output)
        self.assertIn("--- Deception/Integrity Score ---", output)
        self.assertIn("The analysis confirms a high degree of semantic integrity", output)
        self.assertIn("consistent with the input's high 'Justice' value of 0.90.", output)


if __name__ == '__main__':
    unittest.main()
