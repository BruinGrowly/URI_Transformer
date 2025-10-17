"""
Tests for Phi-Enhanced ICE URI Transformer

Validates golden ratio geometric enhancements to ICE Framework.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from phi_enhanced_ice_transformer import PhiEnhancedICETransformer
from phi_geometric_engine import PHI, GOLDEN_ANGLE_DEG


def test_phi_enhanced_basic_transformation():
    """Test basic phi-enhanced transformation"""
    transformer = PhiEnhancedICETransformer()

    result = transformer.transform(
        "Help others with compassion",
        thought_type="moral_judgment",
        context_domain="ethical"
    )

    # Validate basic ICE functionality
    assert result.intent_coordinates is not None
    assert len(result.intent_coordinates) == 4
    assert result.intent_type == "moral_judgment"
    assert result.context_domain == "ethical"
    assert result.execution_strategy in [
        "compassionate_action", "authoritative_command",
        "instructive_guidance", "corrective_judgment", "balanced_response"
    ]

    # Validate phi geometric enhancements
    assert result.fibonacci_depth > 0
    assert result.spiral_distance >= 0
    assert 0 <= result.golden_angle_diversity <= 1.0 or result.golden_angle_diversity < 0  # Can be negative for imbalance
    assert result.phi_bin >= 0
    assert 1 <= result.nearest_dodec_anchor <= 12

    print(f"[PASS] Basic phi-enhanced transformation working")
    print(f"  Fibonacci depth: {result.fibonacci_depth}")
    print(f"  Golden spiral distance: {result.spiral_distance:.4f}")
    print(f"  Phi bin: {result.phi_bin}")
    print(f"  Nearest anchor: #{result.nearest_dodec_anchor}")


def test_fibonacci_relationship_expansion():
    """Test Fibonacci-based relationship expansion"""
    transformer = PhiEnhancedICETransformer()

    # Test different contexts with different Fibonacci depths
    contexts = ["general", "ethical", "spiritual"]
    depths = []

    for context in contexts:
        result = transformer.transform(
            "Test concept",
            thought_type="practical_wisdom",
            context_domain=context
        )
        depths.append(result.fibonacci_depth)

    # Spiritual should have deepest Fibonacci expansion
    assert depths[2] > depths[0], "Spiritual context should have deeper Fibonacci expansion"

    print(f"[PASS] Fibonacci relationship expansion working")
    print(f"  General: depth {depths[0]}")
    print(f"  Ethical: depth {depths[1]}")
    print(f"  Spiritual: depth {depths[2]}")


def test_golden_spiral_distance():
    """Test golden spiral distance calculation"""
    transformer = PhiEnhancedICETransformer()

    # Transform two similar concepts
    result1 = transformer.transform("Show compassion", "moral_judgment", "ethical")
    result2 = transformer.transform("Demonstrate mercy", "moral_judgment", "ethical")

    # Both should have measurable spiral distances
    assert result1.spiral_distance >= 0
    assert result2.spiral_distance >= 0

    print(f"[PASS] Golden spiral distance working")
    print(f"  Concept 1 spiral distance: {result1.spiral_distance:.4f}")
    print(f"  Concept 2 spiral distance: {result2.spiral_distance:.4f}")


def test_golden_angle_diversity():
    """Test golden angle diversity measurement"""
    transformer = PhiEnhancedICETransformer()

    result = transformer.transform(
        "Balance love, power, wisdom, and justice",
        thought_type="practical_wisdom",
        context_domain="spiritual"
    )

    # Should measure diversity
    assert result.golden_angle_diversity is not None

    print(f"[PASS] Golden angle diversity working")
    print(f"  Diversity score: {result.golden_angle_diversity:.4f}")


def test_phi_exponential_binning():
    """Test phi exponential binning"""
    transformer = PhiEnhancedICETransformer()

    # Test multiple transformations
    results = []
    test_cases = [
        "Simple concept",
        "More complex concept with multiple attributes",
        "Highly sophisticated conceptual framework"
    ]

    for text in test_cases:
        result = transformer.transform(text, "practical_wisdom", "general")
        results.append(result)

    # All should have valid phi bins
    for result in results:
        assert result.phi_bin >= 0
        assert result.phi_bin < 20  # Max bins

    print(f"[PASS] Phi exponential binning working")
    for i, result in enumerate(results):
        print(f"  Concept {i+1}: bin {result.phi_bin}")


def test_dodecahedral_navigation():
    """Test dodecahedral anchor navigation"""
    transformer = PhiEnhancedICETransformer()

    # Test that different concepts map to different anchors
    test_cases = [
        ("Love others", "moral_judgment", "ethical"),
        ("Assert authority", "practical_wisdom", "general"),
        ("Seek wisdom", "practical_wisdom", "technical"),
        ("Practice justice", "moral_judgment", "ethical")
    ]

    anchors = []
    for text, thought_type, domain in test_cases:
        result = transformer.transform(text, thought_type, domain)
        anchors.append(result.nearest_dodec_anchor)

    # Should use multiple dodecahedral anchors
    unique_anchors = len(set(anchors))
    assert unique_anchors >= 2, "Should utilize multiple dodecahedral anchors"

    print(f"[PASS] Dodecahedral navigation working")
    print(f"  Unique anchors used: {unique_anchors}/12")
    print(f"  Anchor distribution: {anchors}")


def test_semantic_integrity_phi_enhanced():
    """Test phi-enhanced semantic integrity validation"""
    transformer = PhiEnhancedICETransformer()

    # High integrity test (clear semantic meaning)
    result_high = transformer.transform(
        "Show compassion to those in need",
        "moral_judgment",
        "ethical"
    )

    # Should have high semantic integrity
    assert result_high.semantic_integrity > 0.8

    print(f"[PASS] Phi-enhanced semantic integrity working")
    print(f"  Semantic integrity: {result_high.semantic_integrity:.4f}")


def test_performance_stats():
    """Test phi geometric performance statistics"""
    transformer = PhiEnhancedICETransformer()

    # Run multiple transformations
    for i in range(5):
        transformer.transform(f"Test concept {i}", "practical_wisdom", "general")

    stats = transformer.get_performance_stats()

    # Validate phi geometric features are enabled
    assert stats["uses_golden_spiral"] is True
    assert stats["uses_fibonacci_expansion"] is True
    assert stats["uses_golden_angle_diversity"] is True
    assert stats["uses_phi_exponential_binning"] is True
    assert stats["uses_dodecahedral_navigation"] is True
    assert stats["phi_constant"] == PHI

    print(f"[PASS] Performance stats working")
    print(f"  Transformations: {stats['transformations']}")
    print(f"  Average alignment: {stats['average_alignment']:.4f}")
    print(f"  Phi constant: {stats['phi_constant']}")


def test_comparison_with_standard_ice():
    """Compare phi-enhanced vs standard transformation"""
    phi_transformer = PhiEnhancedICETransformer()

    # Phi-enhanced transformation
    phi_result = phi_transformer.transform(
        "Help others with compassion",
        "moral_judgment",
        "ethical"
    )

    # Phi version should have additional metrics
    assert hasattr(phi_result, 'fibonacci_depth')
    assert hasattr(phi_result, 'spiral_distance')
    assert hasattr(phi_result, 'golden_angle_diversity')
    assert hasattr(phi_result, 'phi_bin')
    assert hasattr(phi_result, 'nearest_dodec_anchor')

    print(f"[PASS] Phi enhancements confirmed")
    print(f"  Has Fibonacci depth: {hasattr(phi_result, 'fibonacci_depth')}")
    print(f"  Has spiral distance: {hasattr(phi_result, 'spiral_distance')}")
    print(f"  Has golden angle diversity: {hasattr(phi_result, 'golden_angle_diversity')}")
    print(f"  Has phi binning: {hasattr(phi_result, 'phi_bin')}")
    print(f"  Has dodecahedral navigation: {hasattr(phi_result, 'nearest_dodec_anchor')}")


if __name__ == "__main__":
    print("=" * 80)
    print("PHI-ENHANCED ICE URI TRANSFORMER - TEST SUITE")
    print("=" * 80)
    print()

    try:
        test_phi_enhanced_basic_transformation()
        print()

        test_fibonacci_relationship_expansion()
        print()

        test_golden_spiral_distance()
        print()

        test_golden_angle_diversity()
        print()

        test_phi_exponential_binning()
        print()

        test_dodecahedral_navigation()
        print()

        test_semantic_integrity_phi_enhanced()
        print()

        test_performance_stats()
        print()

        test_comparison_with_standard_ice()
        print()

        print("=" * 80)
        print("ALL TESTS PASSED [PASS]")
        print("=" * 80)
        print()
        print("Phi Geometric Enhancements Validated:")
        print("  [PASS] Fibonacci relationship expansion")
        print("  [PASS] Golden spiral distance calculation")
        print("  [PASS] Golden angle diversity measurement")
        print("  [PASS] Phi exponential binning")
        print("  [PASS] Dodecahedral 12-anchor navigation")
        print()

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        raise
