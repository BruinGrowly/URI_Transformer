"""
Unit Tests for the Output Generator
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.output_generator import OutputGenerator
from src.data_structures import (
    TruthSenseResult,
    Intent,
    QLAEContext,
    QLAEDomain,
    ExecutionPlan,
    ExecutionStrategy,
    Trajectory
)
from src.phi_geometric_engine import PhiCoordinate

class TestOutputGenerator(unittest.TestCase):
    """A unit test suite for the OutputGenerator."""

    def setUp(self):
        self.generator = OutputGenerator()

    def test_synthesize_output(self):
        """Tests the synthesis of the narrative output."""
        result = TruthSenseResult(
            raw_coord=PhiCoordinate(0, 0, 0, 0),
            aligned_coord=PhiCoordinate(1, 1, 1, 1),
            intent=Intent(purpose="Test Purpose", guiding_principles=["Test Principle"]),
            context=QLAEContext(domains={}, primary_domain=QLAEDomain.ICE, is_valid=True),
            execution=ExecutionPlan(strategy=ExecutionStrategy.COMPASSIONATE_ACTION, magnitude=0.8, description="Test Description"),
            final_output="",
            anchor_distance=0.5,
            harmony_index=0.67,
            semantic_integrity=1.0,
            truth_sense_validation=True,
            deception_score=0.1,
            foundational_principle="Test Principle",
            trajectory=Trajectory(velocity=PhiCoordinate(1, 1, 1, 1), acceleration=2.0)
        )

        output = self.generator.synthesize_output(result)

        self.assertIn("Test Purpose", output)
        self.assertIn("Consciousness", output)
        self.assertIn("Compassionate Action", output)
        self.assertIn("Test Principle", output)
        self.assertIn("harmony index of 0.67", output)
        self.assertIn("acceleration of 2.00", output)
        self.assertIn("deception score of 0.10", output)

if __name__ == '__main__':
    unittest.main()
