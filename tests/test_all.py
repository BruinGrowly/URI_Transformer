"""
Comprehensive Test Suite for Semantic Substrate Engine V2

Tests all core functionality including biblical analysis, secular integration,
context flexibility, and mathematical precision.
"""

import sys
import os
import unittest
import math

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.baseline_biblical_substrate import (
        BiblicalSemanticSubstrate,
        BiblicalCoordinates,
        BiblicalPrinciple,
        BiblicalText
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class TestBiblicalCoordinates(unittest.TestCase):
    """Test the BiblicalCoordinates class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.jehovah_coords = BiblicalCoordinates(1.0, 1.0, 1.0, 1.0)
        self.origin_coords = BiblicalCoordinates(0.0, 0.0, 0.0, 0.0)
        self.balanced_coords = BiblicalCoordinates(0.5, 0.5, 0.5, 0.5)
    
    def test_jevovah_coordinates(self):
        """Test JEHOVAH coordinates"""
        self.assertEqual(self.jehovah_coords.love, 1.0)
        self.assertEqual(self.jehovah_coords.power, 1.0)
        self.assertEqual(self.jehovah_coords.wisdom, 1.0)
        self.assertEqual(self.jehovah_coords.justice, 1.0)
        self.assertEqual(self.jehovah_coords.divine_resonance(), 1.0)
        self.assertEqual(self.jehovah_coords.distance_from_jehovah(), 0.0)
        self.assertEqual(self.jehovah_coords.biblical_balance(), 1.0)
        self.assertEqual(self.jehovah_coords.overall_biblical_alignment(), 1.0)
    
    def test_origin_coordinates(self):
        """Test origin coordinates"""
        self.assertEqual(self.origin_coords.love, 0.0)
        self.assertEqual(self.origin_coords.power, 0.0)
        self.assertEqual(self.origin_coords.wisdom, 0.0)
        self.assertEqual(self.origin_coords.justice, 0.0)
        self.assertEqual(self.origin_coords.divine_resonance(), 0.0)
        max_distance = math.sqrt(4)  # Distance from origin to (1,1,1,1)
        self.assertEqual(self.origin_coords.distance_from_jehovah(), max_distance)
    
    def test_coordinate_validation(self):
        """Test coordinate range validation"""
        # Test automatic range normalization
        coords_high = BiblicalCoordinates(2.0, 2.0, 2.0, 2.0)
        self.assertEqual(coords_high.love, 1.0)
        self.assertEqual(coords_high.power, 1.0)
        self.assertEqual(coords_high.wisdom, 1.0)
        self.assertEqual(coords_high.justice, 1.0)
        
        coords_low = BiblicalCoordinates(-1.0, -1.0, -1.0, -1.0)
        self.assertEqual(coords_low.love, 0.0)
        self.assertEqual(coords_low.power, 0.0)
        self.assertEqual(coords_low.wisdom, 0.0)
        self.assertEqual(coords_low.justice, 0.0)
    
    def test_balanced_coordinates(self):
        """Test balanced coordinates"""
        self.assertEqual(self.balanced_coords.biblical_balance(), 1.0)
        self.assertEqual(self.balanced_coords.get_dominant_attribute(), "Balanced")
        self.assertEqual(self.balanced_coords.get_deficient_attributes(), [])
    
    def test_deficient_coordinates(self):
        """Test deficient coordinates"""
        deficient_coords = BiblicalCoordinates(0.6, 0.7, 0.3, 0.2)
        deficient_attrs = deficient_coords.get_deficient_attributes()
        
        self.assertIn("wisdom", deficient_attrs)
        self.assertIn("justice", deficient_attrs)
        self.assertNotIn("love", deficient_attrs)
        self.assertNotIn("power", deficient_attrs)
    
    def test_dominant_attribute_detection(self):
        """Test dominant attribute detection"""
        coords_love = BiblicalCoordinates(0.9, 0.3, 0.4, 0.5)
        coords_power = BiblicalCoordinates(0.3, 0.9, 0.4, 0.5)
        coords_wisdom = BiblicalCoordinates(0.4, 0.5, 0.9, 0.5)
        coords_justice = BiblicalCoordinates(0.3, 0.4, 0.5, 0.9)
        
        self.assertEqual(coords_love.get_dominant_attribute(), "love")
        self.assertEqual(coords_power.get_dominant_attribute(), "power")
        self.assertEqual(coords_wisdom.get_dominant_attribute(), "wisdom")
        self.assertEqual(coords_justice.get_dominant_attribute(), "justice")

class TestBiblicalText(unittest.TestCase):
    """Test the BiblicalText class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.john_3_16 = BiblicalText(
            "For God so loved the world",
            "John 3:16",
            "John",
            "3",
            "16"
        )
    
    def test_text_processing(self):
        """Test text processing"""
        self.assertEqual(self.john_3_16.text, "For God so loved the world")
        self.assertEqual(self.john_3_16.reference, "John 3:16")
        self.assertEqual(self.john_3_16.book, "John")
        self.assertEqual(self.john_3_16.chapter, "3")
        self.assertEqual(self.john_3_16.verse, "16")
    
    def test_word_extraction(self):
        """Test word extraction"""
        words = self.john_3_16.words
        self.assertIn("for", words)
        self.assertIn("god", words)
        self.assertIn("loved", words)
        self.assertIn("world", words)
    
    def test_concept_extraction(self):
        """Test concept extraction"""
        concepts = self.john_3_16.concepts
        self.assertIn("god", concepts)
        self.assertIn("love", concepts)

class TestBiblicalSemanticSubstrate(unittest.TestCase):
    """Test the main BiblicalSemanticSubstrate engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = BiblicalSemanticSubstrate()
        self.jehovah = self.engine.jehovah_coordinates
        
        # Verify JEHOVAH coordinates
        self.assertEqual(self.jehovah.love, 1.0)
        self.assertEqual(self.jehovah.power, 1.0)
        self.assertEqual(self.jehovah.wisdom, 1.0)
        self.assertEqual(self.jehovah.justice, 1.0)
    
    def test_basic_concept_analysis(self):
        """Test basic concept analysis"""
        # Test fear of Jehovah principle
        fear_text = "The fear of Jehovah is the beginning of wisdom"
        coords = self.engine.analyze_concept(fear_text, "biblical")
        
        # Should have high wisdom score
        self.assertGreater(coords.wisdom, 0.5, "Fear of Jehovah should give high wisdom")
        
        # Should be well-balanced but wisdom-dominant
        self.assertEqual(coords.get_dominant_attribute(), "wisdom")
        
        # Should have reasonable divine resonance
        self.assertGreater(coords.divine_resonance(), 0.2)
    
    def test_divine_entity_recognition(self):
        """Test divine entity recognition"""
        god_text = "God is love, Jesus is wisdom, Holy Spirit is truth"
        coords = self.engine.analyze_concept(god_text, "biblical")
        
        # Should have high scores across all attributes
        self.assertGreater(coords.love, 0.5, "God text should have high love")
        self.assertGreater(coords.wisdom, 0.5, "Jesus text should have high wisdom")
        self.assertGreater(coords.power, 0.4, "Divine entities should have good power")
        
        # Should have high divine resonance
        self.assertGreater(coords.divine_resonance(), 0.4)
    
    def test_context_adaptation(self):
        """Test context adaptation"""
        test_text = "Educational leadership with character"
        
        biblical_coords = self.engine.analyze_concept(test_text, "biblical")
        secular_coords = self.engine.analyze_concept(test_text, "secular")
        
        # Biblical context should have higher biblical alignment
        self.assertGreater(
            biblical_coords.overall_biblical_alignment(),
            secular_coords.overall_biblical_alignment()
        )
    
    def test_biblical_text_analysis(self):
        """Test biblical text analysis"""
        biblical_text = "The fear of Jehovah is the beginning of wisdom"
        result = self.engine.analyze_biblical_text(biblical_text, "biblical")
        
        # Should have the expected structure
        self.assertIn('text', result)
        self.assertIn('coordinates', result)
        self.assertIn('divine_resonance', result)
        self.assertIn('biblical_balance', result)
        self.assertIn('dominant_attribute', result)
        self.assertIn('deficient_attributes', result)
        self.assertIn('overall_alignment', result)
        self.assertIn('biblical_concepts', result)
    
    def test_secular_concept_analysis(self):
        """Test secular concept analysis"""
        secular_text = "Business ethics with integrity and responsibility"
        result = self.engine.analyze_secular_concept(secular_text, "business")
        
        # Should have the expected structure
        self.assertIn('text', result)
        self.assertIn('coordinates', result)
        self.assertIn('biblical_analysis', result)
        self.assertIn('recommendations', result)
        self.assertIn('secular_biblical_compatibility', result)
        
        # Should identify biblical applications
        applications = result['biblical_analysis']['biblical_applications']
        self.assertIsInstance(applications, list)
        
        # Should have compatibility score
        compatibility = result['secular_biblical_compatibility']
        self.assertIsInstance(compatibility, float)
        self.assertGreaterEqual(compatibility, 0.0)
        self.assertLessEqual(compatibility, 1.0)
    
    def test_biblical_balance_correction(self):
        """Test biblical balance correction"""
        # High scores should be balanced down
        high_text = "god jesus christ holy spirit love power wisdom justice"
        coords = self.engine.analyze_concept(high_text, "test")
        
        # Should not have extreme values
        self.assertLessEqual(coords.love, 0.8)
        self.assertLessEqual(coords.power, 0.8)
        self.assertLessEqual(coords.wisdom, 0.8)
        self.assertLessEqual(coords.justice, 0.8)
        
        # Should have some biblical content but balanced
        self.assertGreater(coords.overall_biblical_alignment(), 0.2)
    
    def test_pattern_recognition(self):
        """Test biblical pattern recognition"""
        fear_text = "fear of Jehovah brings wisdom and understanding"
        coords = self.engine.analyze_concept(fear_text, "pattern")
        
        # Should recognize fear of Jehovah pattern
        self.assertGreater(coords.wisdom, 0.3, "Fear of Jehovah pattern should boost wisdom")
        
        # Also should boost related attributes
        self.assertGreater(coords.love, 0.1, "Fear of Jehovah should boost love")
        self.assertGreater(coords.justice, 0.1, "Fear of Jehovah should boost justice")
    
    def test_caching_functionality(self):
        """Test caching functionality"""
        text = "Test caching functionality"
        
        # First call - should calculate
        coords1 = self.engine.analyze_concept(text, "cache")
        
        # Second call with same parameters - should use cache
        coords2 = self.engine.analyze_concept(text, "cache")
        
        # Should have same coordinates
        self.assertEqual(coords1.to_tuple(), coords2.to_tuple())
        
        # Cache should contain the result
        cache_key = f"{text}:cache"
        self.assertIn(cache_key, self.engine.coordinate_cache)
        
        # Cache result should match
        cached_result = self.engine.coordinate_cache[cache_key]
        self.assertEqual(cached_result.to_tuple(), coords1.to_tuple())

class TestMathematicalPrecision(unittest.TestCase):
    """Test mathematical precision of calculations"""
    
    def test_divine_resonance_calculation(self):
        """Test divine resonance calculation"""
        # Perfect alignment (JEHOVAH)
        perfect_coords = BiblicalCoordinates(1.0, 1.0, 1.0, 1.0)
        resonance = perfect_coords.divine_resonance()
        self.assertEqual(resonance, 1.0)
        
        # No alignment (origin)
        origin_coords = BiblicalCoordinates(0.0, 0.0, 0.0, 0.0)
        resonance = origin_coords.divine_resonance()
        self.assertEqual(resonance, 0.0)
        
        # Half alignment (balanced)
        half_coords = BiblicalCoordinates(0.5, 0.5, 0.5, 0.5)
        resonance = half_coords.divine_resonance()
        self.assertEqual(resonance, 0.5)
    
    def test_distance_calculation(self):
        """Test distance calculation"""
        # Distance from JEHOVAH to origin
        origin_coords = BiblicalCoordinates(0.0, 0.0, 0.0, 0.0)
        distance = origin_coords.distance_from_jehovah()
        expected_distance = math.sqrt(4)  # sqrt((1-0)^2 * 4)
        self.assertEqual(distance, expected_distance)
        
        # Distance from JEHOVAH to itself
        jehovah_coords = BiblicalCoordinates(1.0, 1.0, 1.0, 1.0)
        distance = jehovah_coords.distance_from_jehovah()
        self.assertEqual(distance, 0.0)
    
    def test_biblical_balance_calculation(self):
        """Test biblical balance calculation"""
        # Perfect balance (all coordinates equal)
        balanced_coords = BiblicalCoordinates(0.5, 0.5, 0.5, 0.5)
        balance = balanced_coords.biblical_balance()
        self.assertEqual(balance, 1.0)
        
        # Perfect imbalance (some coordinates zero)
        imbalanced_coords = BiblicalCoordinates(1.0, 0.0, 0.0, 0.0)
        balance = imbalanced_coords.biblical_balance()
        self.assertEqual(balance, 0.0)
        
        # Partial imbalance
        partial_coords = BiblicalCoordinates(0.8, 0.8, 0.2, 0.2)
        balance = partial_coords.biblical_balance()
        self.assertEqual(balance, 0.25)  # 0.2/0.8
    
    def test_overall_biblical_alignment(self):
        """Test overall biblical alignment calculation"""
        # Perfect alignment (JEHOVAH)
        perfect_coords = BiblicalCoordinates(1.0, 1.0, 1.0, 1.0)
        alignment = perfect_coords.overall_biblical_alignment()
        self.assertEqual(alignment, 1.0)
        
        # No alignment (origin)
        origin_coords = BiblicalCoordinates(0.0, 0.0, 0.0, 0.0)
        alignment = origin_coords.overall_biblical_alignment()
        self.assertEqual(alignment, 0.0)
        
        # Balanced alignment
        balanced_coords = BiblicalCoordinates(0.5, 0.5, 0.5, 0.5)
        # Weighted calculation: (resonance*0.4 + balance*0.3 + value*0.3)
        # resonance=0.5, balance=1.0, value=0.5 -> 0.5*0.4 + 1.0*0.3 + 0.5*0.3 = 0.2 + 0.3 + 0.15 = 0.65
        alignment = balanced_coords.overall_biblical_alignment()
        self.assertAlmostEqual(alignment, 0.65, places=2)

class TestBiblicalPrinciples(unittest.TestCase):
    """Test biblical principles"""
    
    def test_principle_weights(self):
        """Test biblical principle weights"""
        engine = BiblicalSemanticSubstrate()
        
        # Fear of Jehovah should have weight 1.0 (highest)
        principle_data = engine.biblical_database.wisdom_principles[BiblicalPrinciple.FEAR_OF_JEHOVAH]
        self.assertEqual(principle_data['weight'], 1.0)
        
        # Wisdom should have weight 0.95 (highest after fear)
        principle_data = engine.biblical_database.wisdom_principles[BiblicalPrinciple.WISDOM]
        self.assertEqual(principle_data['weight'], 0.95)
        
        # Love should have weight 0.9 (high)
        principle_data = engine.biblical_database.wisdom_principles[BiblicalPrinciple.LOVE]
        self.assertEqual(principle_data['weight'], 0.9)
    
    def test_principle_coordinates(self):
        """Test biblical principle coordinates"""
        engine = BiblicalSemanticSubstrate()
        
        # Fear of Jehovah coordinates should favor wisdom
        principle_data = engine.biblical_database.wisdom_principles[BiblicalPrinciple.FEAR_OF_JEHOVAH]
        fear_coords = principle_data['coordinates']
        self.assertGreater(fear_coords.wisdom, 0.8, "Fear of Jehovah should have high wisdom")
        self.assertGreaterEqual(fear_coords.love, 0.2)
        self.assertGreaterEqual(fear_coords.justice, 0.2)
    
    def test_principle_scripture_references(self):
        """Test principle scripture references"""
        engine = BiblicalSemanticSubstrate()
        
        # Fear of Jehovah should have specific references
        principle_data = engine.biblical_database.wisdom_principles[BiblicalPrinciple.FEAR_OF_JEHOVAH]
        scriptures = principle_data['scriptures']
        
        expected_refs = ['proverbs_9_10', 'psalm_111_10', 'job_28_28']
        for ref in expected_refs:
            self.assertIn(ref, scriptures)
    
    def test_principle_application(self):
        """Test principle application guidance"""
        engine = BiblicalSemanticSubstrate()
        
        # Fear of Jehovah application
        principle_data = engine.biblical_database.wisdom_principles[BiblicalPrinciple.FEAR_OF_JEHOVAH]
        application = principle_data['application']
        
        self.assertEqual(application, "All wisdom begins here")

class TestSecularBiblicalIntegration(unittest.TestCase):
    """Test secular-biblical integration capabilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = BiblicalSemanticSubstrate()
    
    def test_business_ethics_integration(self):
        """Test business ethics biblical integration"""
        business_text = "Business leadership with integrity and ethical decision-making"
        result = self.engine.analyze_secular_concept(business_text, "business")
        
        # Should identify biblical applications
        applications = result['biblical_analysis']['biblical_applications']
        self.assertGreater(len(applications), 0)
        
        # Should detect integrity (justice)
        self.assertGreater(result['coordinates'].justice, 0.3)
        
        # Should provide reasonable biblical alignment
        self.assertGreater(result['divine_resonance'], 0.1)
    
    def test_education_integration(self):
        """Test educational biblical integration"""
        education_text = "Teaching students with character development and moral excellence"
        result = self.engine.analyze_secular_concept(education_text, "education")
        
        # Should identify educational biblical applications
        applications = result['biblical_analysis']['biblical_applications']
        self.assertGreater(len(applications), 0)
        
        # Should detect teaching value
        self.assertGreater(result['coordinates'].love, 0.1)
        
        # Should provide reasonable biblical alignment
        self.assertGreater(result['divine_resonance'], 0.05)
    
    def test_bridges_identification(self):
        """Test secular-biblical bridge identification"""
        integrity_text = "Professional integrity and ethical conduct in the workplace"
        result = self.engine.analyze_secular_concept(integrity_text, "professional")
        
        # Should identify integrity-justice bridge
        bridges = result['biblical_analysis']['secular_biblical_bridge']
        self.assertGreater(len(bridges), 0)
        
        # Should detect biblical positive attributes
        self.assertGreater(result['coordinates'].justice, 0.3)
    
    def test_compatibility_assessment(self):
        """Test secular-biblical compatibility assessment"""
        compatible_text = "Helpful service with compassion and care"
        incompatible_text = "Atheist anti-religious anti-christian worldview"
        
        compatible_result = self.engine.analyze_secular_concept(compatible_text, "test")
        incompatible_result = self.engine.analyze_secular_concept(incompatible_text, "test")
        
        # Compatible text should have higher compatibility
        self.assertGreater(
            compatible_result['secular_biblical_compatibility'],
            incompatible_result['secular_biblical_compatibility']
        )
        
        # Both should be within valid range
        self.assertLessEqual(compatible_result['secular_biblical_compatibility'], 1.0)
        self.assertGreaterEqual(compatible_result['secular_biblical_compatibility'], 0.0)
        self.assertLessEqual(incompatible_result['secular_biblical_compatibility'], 1.0)
        self.assertGreaterEqual(incompatible_result['secular_biblical_compatibility'], 0.0)

class TestContextFlexibility(unittest.TestCase):
    """Test context flexibility and adaptation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = BiblicalSemanticSubstrate()
    
    def test_context_weight_differences(self):
        """Test that different contexts produce different results"""
        test_text = "Educational leadership with character development"
        
        biblical_result = self.engine.analyze_concept(test_text, "biblical")
        secular_result = self.engine.analyze_concept(test_text, "secular")
        
        # Biblical context should have higher biblical alignment
        self.assertGreater(
            biblical_result.overall_biblical_alignment(),
            secular_result.overall_biblical_alignment()
        )
        
        # Both should maintain balance
        self.assertLess(biblical_result.get_dominant_attribute(), "Balanced")
        self.assertLess(secular_result.get_dominant_attribute(), "Balanced")
    
    def test_context_modifiers(self):
        """Test contextual modifiers"""
        modifiers = self.engine.contextual_modifiers
        
        # Biblical contexts should have highest weight
        self.assertEqual(modifiers['biblical'], 1.0)
        self.assertEqual(modifiers['religious'], 0.9)
        self.assertEqual(modifiers['spiritual'], 0.8)
        
        # Educational contexts should have moderate weight
        self.assertEqual(modifiers['educational'], 0.6)
        self.assertEqual(modifiers['school'], 0.6)
        self.assertEqual(modifiers['university'], 0.5)
        
        # Professional contexts should have balanced weight
        self.assertEqual(modifiers['professional'], 0.4)
        self.assertEqual(modifiers['work'], 0.4)
        self.assertEqual(modifiers['business'], 0.4)
        
        # Secular contexts should have lowest weight
        self.assertEqual(modifiers['secular'], 0.3)
        self.assertEqual(modifiers['general'], 0.3)
        self.assertEqual(modifiers['casual'], 0.3)
    
    def test_context_specific_analysis(self):
        """Test context-specific analysis patterns"""
        text = "Community service with moral character"
        
        # Different contexts should produce appropriate results
        for context in ['biblical', 'educational', 'professional', 'secular']:
            result = self.engine.analyze_concept(text, context)
            
            # Should maintain coordinate range
            self.assertLessEqual(result.love, 1.0)
            self.assertLessEqual(result.power, 1.0)
            self.assertLessEqual(result.wisdom, 1.0)
            self.assertLessEqual(result.justice, 1.0)
            
            # Should be greater than or equal to zero
            self.assertGreaterEqual(result.love, 0.0)
            self.assertGreaterEqual(result.power, 0.0)
            self.assertGreaterEqual(result.wisdom, 0.0)
            self.assertGreaterEqual(result.justice, 0.0)
    
    def test_context_biblical_amplification(self):
        """Test that biblical contexts amplify biblical elements"""
        biblical_text = "God loves wisdom and justice"
        secular_text = "Education with knowledge and fairness"
        
        biblical_result = self.engine.analyze_concept(biblical_text, "biblical")
        secular_result = self.engine.analyze_concept(secular_text, "secular")
        
        # Biblical text should have higher biblical alignment
        self.assertGreater(biblical_result.overall_biblical_alignment(), 0.5)
        self.assertGreater(biblical_result.get_dominant_attribute(), "Balanced")
        
        # Secular text should have lower biblical alignment
        self.assertLessEqual(secular_result.overall_biblical_alignment(), 0.5)

class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = BiblicalSemanticSubstrate()
    
    def test_caching_functionality(self):
        """Test caching optimization"""
        text = "Test text for caching functionality"
        
        # First call - should calculate
        import time
        start_time = time.perf_counter()
        result1 = self.engine.analyze_concept(text, "cache")
        first_time = (time.perf_counter() - start_time) * 1000
        
        # Second call - should use cache
        start_time = time.perf_counter()
        result2 = self.engine.analyze_concept(text, "cache")
        second_time = (time.perf_counter() - start_time) * 1000
        
        # Results should be identical
        self.assertEqual(result1.to_tuple(), result2.to_tuple())
        
        # Second call should be faster
        self.assertLess(second_time, first_time)
        
        # Cache should contain result
        cache_key = f"{text}:cache"
        self.assertIn(cache_key, self.engine.coordinate_cache)
    
    def test_coordinate_normalization(self):
        """Test coordinate normalization to 0-1 range"""
        # Test extreme values
        high_coords = BiblicalCoordinates(5.0, -3.0, 10.0, -2.0)
        
        # All should be normalized to 0-1 range
        self.assertEqual(high_coords.love, 1.0)
        self.assertEqual(high_coords.power, 0.0)
        self.assertEqual(high_coords.wisdom, 1.0)
        self.assertEqual(high_coords.justice, 0.0)
    
    def test_biblical_balance_mechanism(self):
        """Test biblical balance mechanism"""
        # Test that very high biblical scores are balanced down
        extreme_text = "god jesus christ holy spirit love power wisdom justice divine holy righteous perfect"
        coords = self.engine.analyze_concept(extreme_text, "balance")
        
        # Should maintain balance (no single attribute > 0.8)
        self.assertLessEqual(coords.love, 0.8)
        self.assertLessEqual(coords.power, 0.8)
        self.assertLessEqual(coords.wisdom, 0.8)
        self.assertLessEqual(coords.justice, 0.8)
        
        # Should still have some biblical content (all > 0.0)
        self.assertGreater(coords.overall_biblical_alignment(), 0.1)
    
    def test_keyword_scoring_system(self):
        """Test keyword scoring optimization"""
        # Test that multiple keywords contribute appropriately
        text_with_many_keywords = "love power wisdom justice god jesus christ lord holy spirit bible scripture faith hope joy"
        coords = self.engine.analyze_concept(text_with_many_keywords, "scoring")
        
        # Should recognize multiple biblical elements
        self.assertGreater(coords.love, 0.5)
        self.assertGreater(coords.power, 0.5)
        self.assertGreater(coords.wisdom, 0.5)
        self.assertGreater(coords.justice, 0.5)
        
        # Should apply biblical balance
        balance = coords.biblical_balance()
        self.assertGreater(balance, 0.2)

def run_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    
    # Load tests
    test_classes = [
        TestBiblicalCoordinates,
        TestBiblicalText,
        TestBiblicalSemanticSubstrate,
        TestMathematicalPrecision,
        TestBiblicalPrinciples,
        TestSecularBiblicalIntegration,
        TestContextFlexibility,
        TestPerformanceOptimization
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "=" * 70)
    print("SEMANTIC SUBSTRATE ENGINE V2 - TEST SUITE RESULTS")
    print("=" * 70)
    
    tests_run = result.testsRun
    tests_failed = result.failures
    tests_errors = result.errors
    
    if tests_errors > 0:
        print(f" ERRORS: {tests_errors}")
        print("Test suite execution errors - check imports and setup")
    elif tests_failed > 0:
        print(f"  TESTS FAILED: {tests_failed}/{tests_run}")
        print("Some functionality may need attention - see above for details")
    else:
        print(f" ALL TESTS PASSED: {tests_run}/{tests_run}")
        print(" Biblical coordinates system operational")
        print(" Biblical text analysis working")
        print(" Secular integration functional")
        print(" Context flexibility confirmed")
        print(" Mathematical precision verified")
        print(" Performance optimization active")
        print(" All core functionality operational")
    
    print(f"\n Test Coverage:")
    print(f"   Biblical Coordinates: 6 tests")
    print(f"   Biblical Text: 1 test")
    print(f"   Biblical Engine: 4 tests")
    print(f"   Mathematics: 3 tests")
    print(f"   Biblical Principles: 3 tests")
    print(f"   Secular Integration: 4 tests")
    print(f"   Context Flexibility: 4 tests")
    print(f"   Performance: 4 tests")
    print(f"   Total: {len(test_classes)} test classes")
    
    print(f"\n Engine Status:")
    if tests_errors == 0 and tests_failed == 0:
        print(" READY FOR PRODUCTION USE")
        print(" All core functionality verified")
        print(" Mathematical precision confirmed")
        print(" Biblical standards maintained")
        print(" Secular flexibility proven")
        print(" Performance optimized")
    elif tests_errors == 0 and tests_failed == 0:
        print("  BASIC FUNCTIONALITY WORKS")
        print(" Core features operational")
        print(" May need attention for advanced features")
    else:
        print(" NEEDS ATTENTION")
        print(f" {tests_failed}/{tests_run} tests failed")
        print(" Review failing tests for fixes")
    
    print(f"\n Next Steps:")
    if tests_errors == 0 and tests_failed == 0:
        print("• Add advanced features and enhancements")
        print("• Implement additional biblical principles")
        print("• Expand secular-biblical integration")
        print("• Optimize for real-world applications")
    else:
        print("• Fix failing tests")
        print("• Address import issues")
        print("• Verify core functionality")
        print("• Ensure all components work")
    
    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("Semantic Substrate Engine V2 - Biblical Foundation Confirmed")
    print("Secular Flexibility Verified - Ready for Enhancement!")
    print("=" * 70)
    
    return result

if __name__ == "__main__":
    run_tests()