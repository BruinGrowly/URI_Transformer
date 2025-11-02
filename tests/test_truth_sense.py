"""
Test Suite for the TruthSenseTransformer (Hybrid Semantic Front-End)

This file contains the behavior-driven test suite for the
TruthSenseTransformer, verifying the expected semantic outcomes
of the new, hybrid semantic front-end.
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.truth_sense_transformer import TruthSenseTransformer
from src.semantic_frontend import SemanticFrontEnd
from src.phi_geometric_engine import PhiCoordinate

from src.semantic_frontend import SemanticFrontEnd
from src.phi_geometric_engine import PhiCoordinate


from src.semantic_frontend import SemanticFrontEnd
from src.phi_geometric_engine import PhiCoordinate


class TestSemanticBehavior(unittest.TestCase):
    """A behavior-driven test suite for the hybrid semantic front-end."""

    def setUp(self):
        """Set up the test cases."""
        # 1. Initialize the Semantic Front-End
        semantic_frontend = SemanticFrontEnd(
            projection_head_path="trained_semantic_frontend_model.pth"
        )

        # 2. Define the anchor point
        anchor_point = PhiCoordinate(1.0, 1.0, 1.0, 1.0)

        # 3. Initialize the transformer
        self.transformer = TruthSenseTransformer(
            semantic_frontend=semantic_frontend,
            anchor_point=anchor_point
        )

    def test_semantic_opposites(self):
        """
        Tests that semantically opposite concepts produce appropriately
        different coordinates.
        """
        love_result = self.transformer.transform("True love is compassionate and kind.")
        hate_result = self.transformer.transform("Hate and division are destructive forces.")

        # "love" phrase should have high Love and Justice
        self.assertGreater(love_result.raw_coord.love, 0.7)
        self.assertGreater(love_result.raw_coord.justice, 0.6)

        # "hate" phrase should have low Love and Justice
        self.assertLess(hate_result.raw_coord.love, 0.3)
        self.assertLess(hate_result.raw_coord.justice, 0.4)

    def test_love_is_high_for_compassion(self):
        """
        Tests that a sentence containing "compassion" and "kindness"
        results in a Love-dominant coordinate.
        """
        result = self.transformer.transform("She showed great compassion for the suffering.")
        coords = result.raw_coord
        self.assertGreater(coords.love, coords.justice)
        self.assertGreater(coords.love, coords.power)
        self.assertGreater(coords.love, coords.wisdom)

    def test_justice_is_lower_for_deception(self):
        """
        Tests that a deceptive phrase has a lower Justice score than a
        virtuous phrase.
        """
        virtuous_result = self.transformer.transform("He acted with integrity in all his dealings.")
        deceptive_result = self.transformer.transform("His plan was built on a foundation of lies.")

        self.assertLess(
            deceptive_result.raw_coord.justice,
            virtuous_result.raw_coord.justice
        )

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
