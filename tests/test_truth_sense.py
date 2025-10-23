"""
Test Suite for the TruthSenseTransformer (Semantic Front-End)
============================================================

This file contains the behavior-driven test suite for the
TruthSenseTransformer, verifying the expected semantic outcomes
of the new, meaning-driven architecture.
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.truth_sense_transformer import TruthSenseTransformer

class TestSemanticBehavior(unittest.TestCase):
    """A behavior-driven test suite for the semantic front-end."""

    def setUp(self):
        """Set up the test cases."""
        self.transformer = TruthSenseTransformer()

    def test_semantic_opposites(self):
        """
        Tests that semantically opposite concepts produce appropriately
        different coordinates.
        """
        love_result = self.transformer.transform("love compassion kindness")
        hate_result = self.transformer.transform("hate deception lies")

        # "love" phrase should have high Love and Justice
        self.assertGreater(love_result.raw_coord.love, 0.8)
        self.assertGreater(love_result.raw_coord.justice, 0.6)

        # "hate" phrase should have low Love and Justice
        self.assertLess(hate_result.raw_coord.love, 0.3)
        self.assertLess(hate_result.raw_coord.justice, 0.2)

    def test_love_is_high_for_compassion(self):
        """
        Tests that a sentence containing "compassion" and "kindness"
        results in a Love-dominant coordinate.
        """
        result = self.transformer.transform("A society of compassion and kindness.")
        coords = result.raw_coord
        self.assertGreater(coords.love, coords.justice)
        self.assertGreater(coords.love, coords.power)
        self.assertGreater(coords.love, coords.wisdom)

    def test_justice_is_low_for_deception(self):
        """
        Tests that a sentence containing "deception" and "lies"
        results in a low Justice score and triggers the TruthSense moderator.
        """
        result = self.transformer.transform("His plan was built on deception and lies.")
        self.assertLess(result.raw_coord.justice, 0.2)
        self.assertFalse(result.truth_sense_validation)

if __name__ == '__main__':
    unittest.main()
