"""

This file contains the behavior-driven test suite for the
TruthSenseTransformer, verifying the expected semantic outcomes
of the new, meaning-driven architecture.
"""

import unittest
Test Suite for the TruthSenseTransformer (Deep & Generative ICE)

This file contains the test suite for the refactored TruthSenseTransformer,
verifying the deep, integrated ICE pipeline with deterministic tests.
"""

import unittest
from unittest.mock import patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.truth_sense_transformer import TruthSenseTransformer
from src.data_structures import Intent, TruthSenseResult
from src.frameworks import QLAEContext, ExecutionPlan
from src.phi_geometric_engine import PhiCoordinate

class TestDeepICE(unittest.TestCase):
    """Test suite for the deep and generative ICE implementation."""

    def setUp(self):
        """Set up the test cases."""
        self.transformer = TruthSenseTransformer()
        self.high_justice_coord = PhiCoordinate(love=0.8, justice=0.9, power=0.7, wisdom=0.6)
        self.low_justice_coord = PhiCoordinate(love=0.2, justice=0.1, power=0.9, wisdom=0.3)

    @patch('src.truth_sense_transformer.TruthSenseTransformer._generate_deterministic_coord')
    def test_structured_objects(self, mock_generate_coord):
        """Tests that the pipeline produces the correct structured objects."""
        mock_generate_coord.return_value = self.high_justice_coord
        result = self.transformer.transform("Test phrase")

        self.assertIsInstance(result.intent, Intent)
        self.assertIsInstance(result.context, QLAEContext)
        self.assertIsInstance(result.execution, ExecutionPlan)

    @patch('src.truth_sense_transformer.TruthSenseTransformer._generate_deterministic_coord')
    def test_justice_as_moderator(self, mock_generate_coord):
        """Tests that the Justice score correctly moderates the context."""
        # Test with high justice
        mock_generate_coord.return_value = self.high_justice_coord
        high_j_result = self.transformer.transform("A phrase of truth.")
        self.assertTrue(high_j_result.context.is_valid)
        self.assertTrue(high_j_result.truth_sense_validation)

        # Test with low justice
        mock_generate_coord.return_value = self.low_justice_coord
        low_j_result = self.transformer.transform("A phrase of deception.")
        self.assertFalse(low_j_result.context.is_valid)
        self.assertFalse(low_j_result.truth_sense_validation)

    @patch('src.truth_sense_transformer.TruthSenseTransformer._generate_deterministic_coord')
    def test_generative_output(self, mock_generate_coord):
        """Tests that the generative output is correctly synthesized for both valid and invalid contexts."""
        # Test with high justice (valid context)
        mock_generate_coord.return_value = self.high_justice_coord
        high_j_result = self.transformer.transform("A phrase of truth.")
        self.assertIn(high_j_result.context.primary_domain.value.lower(), high_j_result.final_output.lower())
        self.assertNotIn("questionable truth", high_j_result.final_output.lower())

        # Test with low justice (invalid context)
        mock_generate_coord.return_value = self.low_justice_coord
        low_j_result = self.transformer.transform("A phrase of deception.")
        self.assertIn("questionable truth", low_j_result.final_output.lower())
        self.assertNotIn(low_j_result.context.primary_domain.value.lower(), low_j_result.final_output.lower())

if __name__ == '__main__':
    unittest.main()
