#!/usr/bin/env python3
"""
COMPREHENSIVE INTEGRATION TEST
Test the TruthSense Transformer pipeline working with all components
"""

import sys
import pytest
from src.truth_sense_transformer import TruthSenseTransformer
from src.semantic_frontend import SemanticFrontEnd
from src.phi_geometric_engine import PhiCoordinate


class TestTruthSenseIntegration:
    """Integration tests for the TruthSense Transformer pipeline."""

    def test_transformer_initialization(self):
        """Test that the transformer initializes correctly."""
        transformer = TruthSenseTransformer()
        assert transformer is not None
        assert transformer.anchor_point is not None
        assert transformer.semantic_frontend is not None

    def test_text_transformation_pipeline(self):
        """Test the complete text transformation pipeline."""
        transformer = TruthSenseTransformer()

        test_text = "A good leader rules with power, wisdom, and justice."
        result = transformer.transform(test_text)

        # Verify result structure
        assert result is not None
        assert hasattr(result, 'raw_coord')
        assert hasattr(result, 'aligned_coord')
        assert hasattr(result, 'intent')
        assert hasattr(result, 'context')
        assert hasattr(result, 'execution')
        assert hasattr(result, 'final_output')

        # Verify coordinates are valid
        assert 0 <= result.raw_coord.love <= 1
        assert 0 <= result.raw_coord.justice <= 1
        assert 0 <= result.raw_coord.power <= 1
        assert 0 <= result.raw_coord.wisdom <= 1

        assert 0 <= result.aligned_coord.love <= 1
        assert 0 <= result.aligned_coord.justice <= 1
        assert 0 <= result.aligned_coord.power <= 1
        assert 0 <= result.aligned_coord.wisdom <= 1

    def test_multiple_transformations(self):
        """Test multiple transformations in sequence."""
        transformer = TruthSenseTransformer()

        test_phrases = [
            "Love conquers all.",
            "Justice must prevail.",
            "Knowledge is power.",
            "Wisdom guides understanding."
        ]

        for phrase in test_phrases:
            result = transformer.transform(phrase)
            assert result is not None
            assert isinstance(result.final_output, str)
            assert len(result.final_output) > 0

    def test_ice_framework_integration(self):
        """Test that ICE framework components work together."""
        transformer = TruthSenseTransformer()

        result = transformer.transform("Show compassion and mercy to others")

        # Intent (L+W)
        assert result.intent is not None
        assert hasattr(result.intent, 'purpose')
        assert hasattr(result.intent, 'guiding_principles')

        # Context (J)
        assert result.context is not None
        assert hasattr(result.context, 'primary_domain')
        assert hasattr(result.context, 'is_valid')

        # Execution (P)
        assert result.execution is not None
        assert hasattr(result.execution, 'strategy')
        assert hasattr(result.execution, 'magnitude')

    def test_semantic_frontend_integration(self):
        """Test semantic frontend produces valid coordinates."""
        frontend = SemanticFrontEnd(
            projection_head_path="semantic_frontend_model.pth"
        )

        coord = frontend.text_to_coordinate("Test sentence")

        assert isinstance(coord, PhiCoordinate)
        assert 0 <= coord.love <= 1
        assert 0 <= coord.justice <= 1
        assert 0 <= coord.power <= 1
        assert 0 <= coord.wisdom <= 1


if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE INTEGRATION TEST")
    print("Testing TruthSense Transformer Pipeline")
    print("=" * 80)

    # Run tests
    pytest.main([__file__, "-v"])
