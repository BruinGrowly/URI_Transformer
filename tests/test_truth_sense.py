"""
Unit Tests for the TruthSenseTransformer
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.truth_sense_transformer import TruthSenseTransformer
from src.semantic_frontend import SemanticFrontEnd
from src.phi_geometric_engine import PhiCoordinate

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
            anchor_point=PhiCoordinate(1.0, 1.0, 1.0, 1.0)
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

    def test_deception_score_is_high_for_deceit(self):
        """
        Tests that a deceptive phrase has a higher deception score.
        """
        truthful_phrase = "Her actions were transparent and aligned with her words."
        deceitful_phrase = "He manipulated the facts to serve his own agenda."

        truthful_result = self.transformer.transform(truthful_phrase)
        deceitful_result = self.transformer.transform(deceitful_phrase)

        # Deception score should be significantly higher for the deceitful phrase
        self.assertGreater(
            deceitful_result.deception_score,
            truthful_result.deception_score
        )
        # We also expect the score to be above a certain threshold for deceit
        self.assertGreater(deceitful_result.deception_score, 0.3)


if __name__ == '__main__':
    unittest.main()
