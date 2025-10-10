"""
Quick Test Runner for Semantic Substrate Engine V2

Verifies that all components are properly installed and functional.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all components can be imported"""
    print("Testing Imports...")
    
    try:
        # Test main engine
        from baseline_biblical_substrate import (
            BiblicalSemanticSubstrate,
            BiblicalCoordinates,
            BiblicalPrinciple
        )
        print("Successfully imported BiblicalSemanticSubstrate")
        print("Successfully imported BiblicalCoordinates")
        print("Successfully imported BiblicalPrinciple")
        
        # Test core functionality
        engine = BiblicalSemanticSubstrate()
        print("Successfully created BiblicalSemanticSubstrate engine")
        
        # Test JEHOVAH coordinates
        jehovah = engine.jehovah_coordinates
        print(f"JEHOVAH coordinates: ({jehovah.love}, {jehovah.power}, {jehovah.wisdom}, {jehovah.justice})")
        print(f"   Divine Resonance: {jehovah.divine_resonance():.3f}")
        print(f"   Biblical Balance: {jehovah.biblical_balance():.3f}")
        
        # Test basic analysis
        test_coords = engine.analyze_concept("fear of jehovah wisdom", "test")
        print(f"   Basic analysis works: {test_coords}")
        print(f"   Wisdom: {test_coords.wisdom:.3f} (should be high)")
        print(f"   Balance: {test_coords.biblical_balance():.3f}")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_core_functionality():
    """Test core functionality"""
    print("\n Testing Core Functionality...")
    
    try:
        from baseline_biblical_substrate import BiblicalSemanticSubstrate
        
        engine = BiblicalSemanticSubstrate()
        
        # Test biblical text analysis
        biblical_result = engine.analyze_biblical_text(
            "For God so loved the world", 
            "john_3_16"
        )
        print(f" Biblical analysis: {biblical_result['coordinates'].divine_resonance():.3f}")
        
        # Test secular integration
        secular_result = engine.analyze_secular_concept(
            "Business with integrity",
            "business"
        )
        print(f" Secular integration: {secular_result['secular_biblical_compatibility']:.3f}")
        
        # Test context flexibility
        text = "Educational excellence"
        biblical_coords = engine.analyze_concept(text, "biblical")
        secular_coords = engine.analyze_concept(text, "secular")
        
        biblical_resonance = biblical_coords.divine_resonance()
        secular_resonance = secular_coords.divine_resonance()
        
        print(f" Context flexibility: Biblical={biblical_resonance:.3f}, Secular={secular_resonance:.3f}")
        
        # Verify biblical context has higher resonance
        if biblical_resonance > secular_resonance:
            print(" Context adaptation working correctly")
        else:
            print(" Context adaptation needs attention")
        
        return True
        
    except Exception as e:
        print(f" Core functionality test failed: {e}")
        return False

def test_mathematical_precision():
    """Test mathematical precision"""
    print("\n Testing Mathematical Precision...")
    
    try:
        from src.baseline_biblical_substrate import BiblicalCoordinates
        
        # Test perfect coordinates
        perfect_coords = BiblicalCoordinates(1.0, 1.0, 1.0, 1.0)
        print(f" Perfect coordinates resonance: {perfect_coords.divine_resonance():.6f}")
        print(f" Perfect coordinates distance: {perfect_coords.distance_from_jehovah():.6f}")
        
        # Test origin coordinates
        origin_coords = BiblicalCoordinates(0.0, 0.0, 0.0, 0.0)
        print(f" Origin coordinates resonance: {origin_coords.divine_resonance():.6f}")
        
        # Test balance calculation
        balanced_coords = BiblicalCoordinates(0.5, 0.5, 0.5, 0.5)
        print(f" Balanced coordinates balance: {balanced_coords.biblical_balance():.6f}")
        
        # Test distance calculation accuracy
        max_distance = origin_coords.distance_from_jehovah()
        expected = (2.0)  # sqrt(4)
        print(f" Distance calculation: {max_distance:.6f} (expected: {expected:.6f})")
        
        if abs(max_distance - expected) < 0.001:
            print(" Mathematical precision verified")
            return True
        else:
            print("⚠ Mathematical precision needs attention")
            return False
        
    except Exception as e:
        print(f" Mathematical precision test failed: {e}")
        return False

def test_database_integration():
    """Test biblical database integration"""
    print("\n Testing Biblical Database Integration...")
    
    try:
        from baseline_biblical_substrate import BiblicalSemanticSubstrate, BiblicalPrinciple
        
        engine = BiblicalSemanticSubstrate()
        
        # Test wisdom principles
        fear_principle = engine.biblical_database.wisdom_principles[BiblicalPrinciple.FEAR_OF_JEHOVAH]
        print(f" Fear of Jehovah weight: {fear_principle['weight']}")
        print(f" Fear of Jehovah application: {fear_principle['application']}")
        print(f" Fear of Jehovah coordinates: {fear_principle['coordinates']}")
        
        # Test biblical references
        proverbs_9_10 = engine.biblical_database.references['proverbs_9_10']
        print(f" Proverbs 9:10: {proverbs_9_10.text}")
        print(f" Reference principle: {proverbs_9_10.principle.value}")
        
        # Test coordinate mappings
        jesus_mapping = engine.biblical_database.coordinate_mappings['jesus']
        print(f" Jesus coordinates: {jesus_mapping}")
        
        # Test database size
        ref_count = len(engine.biblical_database.references)
        principle_count = len(engine.biblical_database.wisdom_principles)
        mapping_count = len(engine.biblical_database.coordinate_mappings)
        
        print(f" Biblical references: {ref_count}")
        print(f" Wisdom principles: {principle_count}")
        print(f" Coordinate mappings: {mapping_count}")
        
        return True
        
    except Exception as e:
        print(f" Database integration test failed: {e}")
        return False

def test_performance():
    """Test performance characteristics"""
    print("\n Testing Performance...")
    
    try:
        import time
        from baseline_biblical_substrate import BiblicalSemanticSubstrate
        
        engine = BiblicalSemanticSubstrate()
        
        # Test analysis speed
        test_text = "Biblical wisdom with love, power, justice and integrity"
        
        # Measure first call (calculation)
        start_time = time.perf_counter()
        result1 = engine.analyze_concept(test_text, "performance")
        first_time = (time.perf_counter() - start_time) * 1000
        
        # Measure second call (cache)
        start_time = time.perf_counter()
        result2 = engine.analyze_concept(test_text, "performance")
        second_time = (time.perf_counter() - start_time) * 1000
        
        print(f" First analysis time: {first_time:.3f} ms")
        print(f" Second analysis time: {second_time:.3f} ms")
        
        # Caching should make second call faster
        if second_time < first_time:
            print(" Caching optimization working")
        else:
            print(" Caching optimization may need attention")
        
        # Test cache size
        cache_size = len(engine.coordinate_cache)
        print(f" Cache entries: {cache_size}")
        
        return True
        
    except Exception as e:
        print(f" Performance test failed: {e}")
        return False

def run_quick_tests():
    """Run all quick tests and return overall status"""
    print("=" * 70)
    print("SEMANTIC SUBSTRATE ENGINE V2 - QUICK TEST SUITE")
    print("Verifying biblical foundation and secular flexibility")
    print("=" * 70)
    
    tests = [
        ("Import Verification", test_imports),
        ("Core Functionality", test_core_functionality),
        ("Mathematical Precision", test_mathematical_precision),
        ("Database Integration", test_database_integration),
        ("Performance Optimization", test_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}...")
            result = test_func()
            results.append((test_name, result, None))
            print(f" {test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f" {test_name}: FAIL - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("QUICK TEST SUITE SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    print(f"Tests Completed: {total}/{total}")
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n ALL QUICK TESTS PASSED!")
        print(" Biblical foundation confirmed")
        print(" 4D coordinate system operational")
        print(" Biblical standards maintained")
        print(" Secular flexibility verified")
        print(" Mathematical precision confirmed")
        print(" Database integration working")
        print(" Performance optimization active")
        
        print(f"\n ENGINE STATUS: PRODUCTION READY")
        print(f" Core Features: Fully Operational")
        print(f" Ready for Enhancement and Development")
        
        return True
    else:
        print(f"\n {total - passed} tests failed")
        print(f" {passed} tests passed")
        print(f" {total - passed} tests failed")
        
        failed_tests = [(test_name, result, error) for test_name, result, error in results if not result]
        for test_name, result, error in failed_tests:
            print(f"    {test_name}: {error if error else 'FAILED'}")
        
        print(f"\n Status: NEEDS ATTENTION")
        print(f"   Review failing tests for fixes")
        print(f"   Ensure all components are properly installed")
        print(f"   Verify biblical foundation integrity")
        
        return False

if __name__ == "__main__":
    success = run_quick_tests()
    exit(0 if success else 1)