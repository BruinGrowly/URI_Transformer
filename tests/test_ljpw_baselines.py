"""
Tests for LJPW Mathematical Baselines

Validates the mathematical foundations and calculations of the LJPW framework.
"""

import unittest
import math
from src.ljpw_baselines import (
    LJPWBaselines,
    NumericalEquivalents,
    ReferencePoints,
    get_numerical_equivalents,
    get_reference_points
)
from src.phi_geometric_engine import PhiCoordinate


class TestNumericalEquivalents(unittest.TestCase):
    """Test fundamental numerical constants"""

    def test_love_constant(self):
        """Test Love constant (Golden ratio inverse)"""
        ne = NumericalEquivalents()
        expected = (math.sqrt(5) - 1) / 2
        self.assertAlmostEqual(ne.L, expected, places=6)
        self.assertAlmostEqual(ne.L, 0.618034, places=5)

    def test_justice_constant(self):
        """Test Justice constant (Pythagorean ratio)"""
        ne = NumericalEquivalents()
        expected = math.sqrt(2) - 1
        self.assertAlmostEqual(ne.J, expected, places=6)
        self.assertAlmostEqual(ne.J, 0.414214, places=5)

    def test_power_constant(self):
        """Test Power constant (Exponential base)"""
        ne = NumericalEquivalents()
        expected = math.e - 2
        self.assertAlmostEqual(ne.P, expected, places=6)
        self.assertAlmostEqual(ne.P, 0.718282, places=5)

    def test_wisdom_constant(self):
        """Test Wisdom constant (Information unit)"""
        ne = NumericalEquivalents()
        expected = math.log(2)
        self.assertAlmostEqual(ne.W, expected, places=6)
        self.assertAlmostEqual(ne.W, 0.693147, places=5)


class TestReferencePoints(unittest.TestCase):
    """Test reference points in LJPW space"""

    def test_anchor_point(self):
        """Test Anchor Point (perfect ideal)"""
        rp = ReferencePoints()
        self.assertEqual(rp.ANCHOR_POINT, (1.0, 1.0, 1.0, 1.0))

    def test_natural_equilibrium(self):
        """Test Natural Equilibrium (optimal balance)"""
        rp = ReferencePoints()
        ne = rp.NATURAL_EQUILIBRIUM
        self.assertAlmostEqual(ne[0], 0.618034, places=5)  # Love
        self.assertAlmostEqual(ne[1], 0.414214, places=5)  # Justice
        self.assertAlmostEqual(ne[2], 0.718282, places=5)  # Power
        self.assertAlmostEqual(ne[3], 0.693147, places=5)  # Wisdom


class TestEffectiveDimensions(unittest.TestCase):
    """Test coupling-adjusted effective dimensions"""

    def test_effective_dimensions_no_love(self):
        """Test effective dimensions with zero Love"""
        L, J, P, W = 0.0, 0.5, 0.5, 0.5
        eff = LJPWBaselines.effective_dimensions(L, J, P, W)

        self.assertEqual(eff['effective_L'], 0.0)
        self.assertEqual(eff['effective_J'], 0.5)  # No amplification
        self.assertEqual(eff['effective_P'], 0.5)  # No amplification
        self.assertEqual(eff['effective_W'], 0.5)  # No amplification

    def test_effective_dimensions_high_love(self):
        """Test effective dimensions with high Love"""
        L, J, P, W = 1.0, 0.5, 0.5, 0.5
        eff = LJPWBaselines.effective_dimensions(L, J, P, W)

        self.assertEqual(eff['effective_L'], 1.0)
        self.assertAlmostEqual(eff['effective_J'], 0.5 * (1 + 1.4 * 1.0))  # 1.2
        self.assertAlmostEqual(eff['effective_P'], 0.5 * (1 + 1.3 * 1.0))  # 1.15
        self.assertAlmostEqual(eff['effective_W'], 0.5 * (1 + 1.5 * 1.0))  # 1.25

    def test_love_multiplier_effect(self):
        """Test Love's force multiplier effect"""
        # Test at different Love levels
        test_cases = [
            (0.0, 1.00),  # No amplification
            (0.3, 1.42),  # ~42% boost to Justice
            (0.6, 1.84),  # ~84% boost to Justice
            (0.9, 2.26),  # ~126% boost to Justice
        ]

        for love, expected_j_multiplier in test_cases:
            J = 1.0  # Use 1.0 for easy multiplication check
            eff = LJPWBaselines.effective_dimensions(love, J, 0.5, 0.5)
            actual_multiplier = eff['effective_J'] / J
            self.assertAlmostEqual(actual_multiplier, expected_j_multiplier, places=1)


class TestMixingAlgorithms(unittest.TestCase):
    """Test mixing algorithms for LJPW dimensions"""

    def test_harmonic_mean_balanced(self):
        """Test harmonic mean with balanced dimensions"""
        L, J, P, W = 0.5, 0.5, 0.5, 0.5
        result = LJPWBaselines.harmonic_mean(L, J, P, W)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_harmonic_mean_weak_link(self):
        """Test harmonic mean with one weak dimension"""
        L, J, P, W = 0.9, 0.9, 0.9, 0.1  # Weak Wisdom
        result = LJPWBaselines.harmonic_mean(L, J, P, W)
        # Harmonic mean should be dominated by weakest link
        self.assertLess(result, 0.3)

    def test_harmonic_mean_zero_dimension(self):
        """Test harmonic mean with zero dimension"""
        L, J, P, W = 0.0, 0.5, 0.5, 0.5
        result = LJPWBaselines.harmonic_mean(L, J, P, W)
        self.assertEqual(result, 0.0)

    def test_geometric_mean_balanced(self):
        """Test geometric mean with balanced dimensions"""
        L, J, P, W = 0.5, 0.5, 0.5, 0.5
        result = LJPWBaselines.geometric_mean(L, J, P, W)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_geometric_mean_multiplicative(self):
        """Test geometric mean multiplicative property"""
        L, J, P, W = 0.8, 0.8, 0.8, 0.8
        result = LJPWBaselines.geometric_mean(L, J, P, W)
        expected = 0.8  # All equal
        self.assertAlmostEqual(result, expected, places=6)

    def test_coupling_aware_sum_no_love(self):
        """Test coupling-aware sum with zero Love"""
        L, J, P, W = 0.0, 0.5, 0.5, 0.5
        result = LJPWBaselines.coupling_aware_sum(L, J, P, W)
        # Should be weighted sum without amplification
        expected = 0.35 * 0.0 + 0.25 * 0.5 + 0.20 * 0.5 + 0.20 * 0.5
        self.assertAlmostEqual(result, expected, places=6)

    def test_coupling_aware_sum_high_love(self):
        """Test coupling-aware sum with high Love (can exceed 1.0)"""
        L, J, P, W = 1.0, 1.0, 1.0, 1.0
        result = LJPWBaselines.coupling_aware_sum(L, J, P, W)
        # Should exceed 1.0 due to coupling amplification
        self.assertGreater(result, 1.0)

    def test_harmony_index_at_anchor(self):
        """Test harmony index at Anchor Point"""
        L, J, P, W = 1.0, 1.0, 1.0, 1.0
        result = LJPWBaselines.harmony_index(L, J, P, W)
        # At anchor point, distance = 0, so harmony = 1/(1+0) = 1.0
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_harmony_index_far_from_anchor(self):
        """Test harmony index far from Anchor Point"""
        L, J, P, W = 0.0, 0.0, 0.0, 0.0
        result = LJPWBaselines.harmony_index(L, J, P, W)
        # Far from anchor, harmony should be low
        self.assertLess(result, 0.5)

    def test_composite_score_range(self):
        """Test composite score with various inputs"""
        test_cases = [
            (0.2, 0.2, 0.2, 0.2),  # Low scores
            (0.5, 0.5, 0.5, 0.5),  # Medium scores
            (0.8, 0.8, 0.8, 0.8),  # High scores
        ]

        for L, J, P, W in test_cases:
            result = LJPWBaselines.composite_score(L, J, P, W)
            self.assertGreater(result, 0.0)
            # Can exceed 1.0 due to coupling effects
            self.assertLess(result, 2.0)


class TestDistanceMetrics(unittest.TestCase):
    """Test distance calculations"""

    def test_distance_from_anchor_at_anchor(self):
        """Test distance from anchor when at anchor point"""
        L, J, P, W = 1.0, 1.0, 1.0, 1.0
        result = LJPWBaselines.distance_from_anchor(L, J, P, W)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_distance_from_anchor_at_origin(self):
        """Test distance from anchor at origin"""
        L, J, P, W = 0.0, 0.0, 0.0, 0.0
        result = LJPWBaselines.distance_from_anchor(L, J, P, W)
        expected = math.sqrt(4.0)  # sqrt(1^2 + 1^2 + 1^2 + 1^2)
        self.assertAlmostEqual(result, expected, places=6)

    def test_distance_from_natural_equilibrium_at_ne(self):
        """Test distance from NE when at Natural Equilibrium"""
        ne = ReferencePoints.NATURAL_EQUILIBRIUM
        L, J, P, W = ne[0], ne[1], ne[2], ne[3]
        result = LJPWBaselines.distance_from_natural_equilibrium(L, J, P, W)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_distance_from_natural_equilibrium_interpretation(self):
        """Test distance interpretation ranges"""
        # Near-optimal: d < 0.2
        ne = ReferencePoints.NATURAL_EQUILIBRIUM
        L, J, P, W = ne[0] + 0.05, ne[1] + 0.05, ne[2] + 0.05, ne[3] + 0.05
        result = LJPWBaselines.distance_from_natural_equilibrium(L, J, P, W)
        self.assertLess(result, 0.2)  # Near-optimal


class TestFullDiagnostic(unittest.TestCase):
    """Test full diagnostic functionality"""

    def test_full_diagnostic_structure(self):
        """Test full diagnostic returns expected structure"""
        L, J, P, W = 0.792, 0.843, 0.940, 0.724
        diagnostic = LJPWBaselines.full_diagnostic(L, J, P, W)

        # Check structure
        self.assertIn('coordinates', diagnostic)
        self.assertIn('effective_dimensions', diagnostic)
        self.assertIn('distances', diagnostic)
        self.assertIn('metrics', diagnostic)

        # Check coordinates
        self.assertEqual(diagnostic['coordinates']['L'], L)
        self.assertEqual(diagnostic['coordinates']['J'], J)
        self.assertEqual(diagnostic['coordinates']['P'], P)
        self.assertEqual(diagnostic['coordinates']['W'], W)

        # Check effective dimensions
        self.assertIn('effective_L', diagnostic['effective_dimensions'])
        self.assertIn('effective_J', diagnostic['effective_dimensions'])
        self.assertIn('effective_P', diagnostic['effective_dimensions'])
        self.assertIn('effective_W', diagnostic['effective_dimensions'])

        # Check distances
        self.assertIn('from_anchor', diagnostic['distances'])
        self.assertIn('from_natural_equilibrium', diagnostic['distances'])

        # Check metrics
        self.assertIn('harmonic_mean', diagnostic['metrics'])
        self.assertIn('geometric_mean', diagnostic['metrics'])
        self.assertIn('coupling_aware_sum', diagnostic['metrics'])
        self.assertIn('harmony_index', diagnostic['metrics'])
        self.assertIn('composite_score', diagnostic['metrics'])


class TestSuggestImprovements(unittest.TestCase):
    """Test improvement suggestions"""

    def test_suggest_improvements_structure(self):
        """Test suggest_improvements returns expected structure"""
        L, J, P, W = 0.5, 0.3, 0.6, 0.7
        suggestions = LJPWBaselines.suggest_improvements(L, J, P, W)

        self.assertIn('primary_focus', suggestions)
        self.assertIn('primary_focus_key', suggestions)
        self.assertIn('distance_from_optimal', suggestions)
        self.assertIn('current_value', suggestions)
        self.assertIn('target_value', suggestions)
        self.assertIn('all_priorities', suggestions)

    def test_suggest_improvements_identifies_weakest(self):
        """Test that suggestions identify the weakest dimension"""
        # Make Justice significantly lower than NE
        L, J, P, W = 0.618, 0.1, 0.718, 0.693  # Low Justice
        suggestions = LJPWBaselines.suggest_improvements(L, J, P, W)

        # Justice should be primary focus (furthest from NE)
        self.assertEqual(suggestions['primary_focus'], 'Justice')
        self.assertEqual(suggestions['primary_focus_key'], 'J')


class TestPhiCoordinateIntegration(unittest.TestCase):
    """Test PhiCoordinate integration with baselines"""

    def test_phicoordinate_effective_dimensions(self):
        """Test PhiCoordinate.get_effective_dimensions()"""
        coord = PhiCoordinate(love=0.8, justice=0.6, power=0.7, wisdom=0.5)
        eff = coord.get_effective_dimensions()

        self.assertIn('effective_L', eff)
        self.assertIn('effective_J', eff)
        self.assertIn('effective_P', eff)
        self.assertIn('effective_W', eff)

        # Check Love amplification
        self.assertEqual(eff['effective_L'], 0.8)
        self.assertGreater(eff['effective_J'], 0.6)  # Amplified by Love

    def test_phicoordinate_harmonic_mean(self):
        """Test PhiCoordinate.harmonic_mean()"""
        coord = PhiCoordinate(love=0.5, justice=0.5, power=0.5, wisdom=0.5)
        result = coord.harmonic_mean()
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_phicoordinate_geometric_mean(self):
        """Test PhiCoordinate.geometric_mean()"""
        coord = PhiCoordinate(love=0.8, justice=0.8, power=0.8, wisdom=0.8)
        result = coord.geometric_mean()
        self.assertAlmostEqual(result, 0.8, places=6)

    def test_phicoordinate_coupling_aware_sum(self):
        """Test PhiCoordinate.coupling_aware_sum()"""
        coord = PhiCoordinate(love=1.0, justice=1.0, power=1.0, wisdom=1.0)
        result = coord.coupling_aware_sum()
        self.assertGreater(result, 1.0)  # Should exceed 1.0 due to coupling

    def test_phicoordinate_harmony_index(self):
        """Test PhiCoordinate.harmony_index()"""
        coord = PhiCoordinate(love=1.0, justice=1.0, power=1.0, wisdom=1.0)
        result = coord.harmony_index()
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_phicoordinate_composite_score(self):
        """Test PhiCoordinate.composite_score()"""
        coord = PhiCoordinate(love=0.8, justice=0.8, power=0.8, wisdom=0.8)
        result = coord.composite_score()
        self.assertGreater(result, 0.0)

    def test_phicoordinate_distance_from_anchor(self):
        """Test PhiCoordinate.distance_from_anchor()"""
        coord = PhiCoordinate(love=1.0, justice=1.0, power=1.0, wisdom=1.0)
        result = coord.distance_from_anchor()
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_phicoordinate_distance_from_ne(self):
        """Test PhiCoordinate.distance_from_natural_equilibrium()"""
        ne = ReferencePoints.NATURAL_EQUILIBRIUM
        coord = PhiCoordinate(love=ne[0], justice=ne[1], power=ne[2], wisdom=ne[3])
        result = coord.distance_from_natural_equilibrium()
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_phicoordinate_full_diagnostic(self):
        """Test PhiCoordinate.full_diagnostic()"""
        coord = PhiCoordinate(love=0.7, justice=0.6, power=0.8, wisdom=0.5)
        diagnostic = coord.full_diagnostic()

        self.assertIn('coordinates', diagnostic)
        self.assertIn('effective_dimensions', diagnostic)
        self.assertIn('distances', diagnostic)
        self.assertIn('metrics', diagnostic)

    def test_phicoordinate_suggest_improvements(self):
        """Test PhiCoordinate.suggest_improvements()"""
        coord = PhiCoordinate(love=0.5, justice=0.3, power=0.6, wisdom=0.7)
        suggestions = coord.suggest_improvements()

        self.assertIn('primary_focus', suggestions)
        self.assertIn('all_priorities', suggestions)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions"""

    def test_get_numerical_equivalents(self):
        """Test get_numerical_equivalents() function"""
        ne = get_numerical_equivalents()
        self.assertIsInstance(ne, NumericalEquivalents)
        self.assertAlmostEqual(ne.L, 0.618034, places=5)

    def test_get_reference_points(self):
        """Test get_reference_points() function"""
        rp = get_reference_points()
        self.assertIsInstance(rp, ReferencePoints)
        self.assertEqual(rp.ANCHOR_POINT, (1.0, 1.0, 1.0, 1.0))


if __name__ == '__main__':
    unittest.main()
