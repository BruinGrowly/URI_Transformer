"""
Unit Tests for the TruthSenseTransformer
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.truth_sense_transformer import TruthSenseTransformer
from src.semantic_frontend import SemanticFrontEnd
from src.phi_geometric_engine import PhiCoordinate
from src.data_structures import TruthSenseResult

class TestTruthSenseTransformer(unittest.TestCase):
    """A unit test suite for the TruthSenseTransformer."""

    def setUp(self):
        # This is a mock front-end that returns predictable coordinates.
        class MockSemanticFrontEnd:
            def text_to_coordinate(self, text: str) -> PhiCoordinate:
                if "love" in text:
                    return PhiCoordinate(1.8, 1.4, 1.0, 1.6)
                elif "hate" in text:
                    return PhiCoordinate(0.2, 0.4, 1.6, 0.6)
                else:
                    return PhiCoordinate(1.0, 1.0, 1.0, 1.0)

        self.transformer = TruthSenseTransformer(
            semantic_frontend=MockSemanticFrontEnd(),
            anchor_point=PhiCoordinate(1.0, 1.0, 1.0, 1.0),
            mixing_method="linear" # Default to linear for existing tests
        )

    def test_semantic_opposites(self):
        """Tests that semantically opposite concepts are handled differently."""
        love_result = self.transformer.transform("love")
        hate_result = self.transformer.transform("hate")

        self.assertGreater(love_result.harmony_index, hate_result.harmony_index)
        self.assertLess(love_result.deception_score, hate_result.deception_score)

    def test_deception_score(self):
        """Tests the deception score calculation."""
        # High justice should result in a low deception score
        high_justice_coord = PhiCoordinate(1.0, 1.8, 1.0, 1.0)
        deception_score_low = self.transformer.calculate_deception_score(high_justice_coord.justice)
        self.assertAlmostEqual(deception_score_low, 0.0)

        # Low justice should result in a high deception score
        low_justice_coord = PhiCoordinate(1.0, 0.2, 1.0, 1.0)
        deception_score_high = self.transformer.calculate_deception_score(low_justice_coord.justice)
        self.assertAlmostEqual(deception_score_high, 1.6)

    def test_generate_from_recipe(self):
        """Tests the generate_from_recipe method with different mixing methods."""
        test_recipe = {'love': 1, 'justice': 2, 'power': 3, 'wisdom': 4}

        # Test linear mixing
        linear_result = self.transformer.generate_from_recipe(test_recipe, mixing_method="linear")
        self.assertIsInstance(linear_result, TruthSenseResult)
        self.assertIsInstance(linear_result.raw_coord, PhiCoordinate)
        self.assertAlmostEqual(linear_result.raw_coord.love, 0.1)
        self.assertAlmostEqual(linear_result.raw_coord.justice, 0.2)
        self.assertAlmostEqual(linear_result.raw_coord.power, 0.3)
        self.assertAlmostEqual(linear_result.raw_coord.wisdom, 0.4)

        # Test quadratic mixing
        quadratic_result = self.transformer.generate_from_recipe(test_recipe, mixing_method="quadratic")
        self.assertIsInstance(quadratic_result, TruthSenseResult)
        self.assertIsInstance(quadratic_result.raw_coord, PhiCoordinate)
        # Quadratic should emphasize dominant components more than linear
        self.assertGreater(quadratic_result.raw_coord.wisdom, linear_result.raw_coord.wisdom)
        self.assertLess(quadratic_result.raw_coord.love, linear_result.raw_coord.love)

        # Test golden mixing
        golden_result = self.transformer.generate_from_recipe(test_recipe, mixing_method="golden")
        self.assertIsInstance(golden_result, TruthSenseResult)
        self.assertIsInstance(golden_result.raw_coord, PhiCoordinate)
        # Golden should also emphasize dominant components, potentially differently from quadratic
        self.assertGreater(golden_result.raw_coord.wisdom, linear_result.raw_coord.wisdom)
        self.assertLess(golden_result.raw_coord.love, linear_result.raw_coord.love)

        # Ensure different mixing methods produce different raw coordinates for non-uniform recipes
        self.assertFalse(np.array_equal(linear_result.raw_coord.to_numpy(), quadratic_result.raw_coord.to_numpy()))
        self.assertFalse(np.array_equal(linear_result.raw_coord.to_numpy(), golden_result.raw_coord.to_numpy()))
        self.assertFalse(np.array_equal(quadratic_result.raw_coord.to_numpy(), golden_result.raw_coord.to_numpy()))

        # Test a zero-weight recipe (should default to neutral)
        zero_recipe = {}
        zero_result = self.transformer.generate_from_recipe(zero_recipe)
        self.assertAlmostEqual(zero_result.raw_coord.love, 0.25)
        self.assertAlmostEqual(zero_result.raw_coord.justice, 0.25)
        self.assertAlmostEqual(zero_result.raw_coord.power, 0.25)
        self.assertAlmostEqual(zero_result.raw_coord.wisdom, 0.25)

        # Test with a balanced recipe, all methods should yield the same result
        balanced_recipe = {'love': 1, 'justice': 1, 'power': 1, 'wisdom': 1}
        linear_balanced = self.transformer.generate_from_recipe(balanced_recipe, mixing_method="linear")
        quadratic_balanced = self.transformer.generate_from_recipe(balanced_recipe, mixing_method="quadratic")
        golden_balanced = self.transformer.generate_from_recipe(balanced_recipe, mixing_method="golden")

        self.assertTrue(np.array_equal(linear_balanced.raw_coord.to_numpy(), quadratic_balanced.raw_coord.to_numpy()))
        self.assertTrue(np.array_equal(linear_balanced.raw_coord.to_numpy(), golden_balanced.raw_coord.to_numpy()))

        # Check general properties for one result (e.g., linear_result)
        self.assertGreaterEqual(linear_result.aligned_coord.love, 0)
        self.assertLessEqual(linear_result.aligned_coord.love, 2)
        self.assertGreaterEqual(linear_result.harmony_index, 0)
        self.assertLessEqual(linear_result.harmony_index, 1)
        self.assertIsInstance(linear_result.final_output, str)
        self.assertGreater(len(linear_result.final_output), 0)

    def test_transform_code(self):
        """Tests the transform_code method with a sample code snippet."""
        code_snippet = """
def my_function(a, b):
    result = a + b
    return result
"""
        result = self.transformer.transform_code(code_snippet)

        self.assertIsInstance(result, TruthSenseResult)
        self.assertIsInstance(result.raw_coord, PhiCoordinate)

        # Check if coordinates are within the expected range [0, 1] (after universal_semantic_mix normalization)
        self.assertGreaterEqual(result.raw_coord.love, 0)
        self.assertLessEqual(result.raw_coord.love, 1)
        self.assertGreaterEqual(result.raw_coord.justice, 0)
        self.assertLessEqual(result.raw_coord.justice, 1)
        self.assertGreaterEqual(result.raw_coord.power, 0)
        self.assertLessEqual(result.raw_coord.power, 1)
        self.assertGreaterEqual(result.raw_coord.wisdom, 0)
        self.assertLessEqual(result.raw_coord.wisdom, 1)

        # Check that final output is generated
        self.assertIsInstance(result.final_output, str)
        self.assertGreater(len(result.final_output), 0)

        # Test with a code snippet that should be power-dominated
        power_code = """
for i in range(10):
    print(i)
"""
        power_result = self.transformer.transform_code(power_code)
        self.assertGreater(power_result.raw_coord.power, power_result.raw_coord.justice)

        # Test with a code snippet that should be justice-dominated (structure)
        justice_code = """
class MyData:
    def __init__(self, value):
        self.value = value
"""
        justice_result = self.transformer.transform_code(justice_code)
        self.assertGreater(justice_result.raw_coord.justice, justice_result.raw_coord.power)

if __name__ == '__main__':
    unittest.main()
