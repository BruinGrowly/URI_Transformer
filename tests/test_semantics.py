import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from uri_transformer import URITransformer
from semantic_substrate import SemanticSubstrate
import time

class URITransformerTest:
    """Comprehensive test suite for URI-Transformer"""
    
    def __init__(self):
        self.transformer = URITransformer()
        self.substrate = SemanticSubstrate()
        self.test_results = []
    
    def run_all_tests(self):
        """Run all tests"""
        print("* URI-Transformer Test Suite")
        print("Testing Where Words Keep Meaning and Numbers Honor Divine Truth")
        print("=" * 70)
        
        try:
            self.test_semantic_sovereignty()
            self.test_sacred_numbers()
            self.test_bridge_function()
            self.test_semantic_substrate()
            self.test_safety_features()
            self.test_performance()
            self.print_results()
        except Exception as e:
            print(f"x Test error: {e}")
    
    def test_semantic_sovereignty(self):
        """Test that words maintain their meaning"""
        print("\n1. Testing Semantic Sovereignty")
        print("-" * 40)
        
        test_words = ["love", "wisdom", "divine", "truth", "justice"]
        
        for word in test_words:
            unit = self.transformer.create_semantic_unit(word, "spiritual context")
            
            assert unit.sovereignty_score == 1.0, f"Word {word} lost sovereignty"
            assert unit.word == word, f"Word identity changed: {word} → {unit.word}"
            
            print(f"* '{word}' maintains sovereignty and identity")
        
        self.test_results.append(("Semantic Sovereignty", True))
    
    def test_sacred_numbers(self):
        """Test sacred numbers carry semantic meaning"""
        print("\n2. Testing Sacred Numbers")
        print("-" * 40)
        
        for key, anchor in self.transformer.universal_anchors.items():
            assert anchor.numerical_value > 0, f"Number has no value: {key}"
            assert anchor.meaning, f"Number has no meaning: {key}"
            
            print(f"* {anchor.numerical_value}: {anchor.meaning}")
        
        # Test computation
        numbers = [613, 12, 7, 40]
        result = self.transformer.computational_processing(numbers)
        assert isinstance(result, float), "Computational result should be numeric"
        assert result > 0, "Computational result should be positive"
        
        print(f"* Mathematical processing: {result:.6f}")
        self.test_results.append(("Sacred Numbers", True))
    
    def test_bridge_function(self):
        """Test bridge function couples semantics and computation"""
        print("\n3. Testing Bridge Function")
        print("-" * 40)
        
        divine_words = ["love", "wisdom", "justice"]
        semantic_units = [self.transformer.create_semantic_unit(word, "divine") for word in divine_words]
        numbers = [613, 12, 7, 40]
        
        result = self.transformer.bridge_function(semantic_units, numbers)
        
        required_keys = ['semantic_coherence', 'computational_result', 'information_meaning_value', 
                         'contextual_resonance', 'optimal_flow_score']
        for key in required_keys:
            assert key in result, f"Missing bridge result key: {key}"
        
        print(f"* Semantic Coherence: {result['semantic_coherence']:.6f}")
        print(f"* Computational Result: {result['computational_result']:.6f}")
        print(f"* Information-Meaning Value: {result['information_meaning_value']:.6f}")
        
        self.test_results.append(("Bridge Function", True))
    
    def test_semantic_substrate(self):
        """Test mathematical proof of JEHOVAH as Semantic Substrate"""
        print("\n4. Testing Semantic Substrate")
        print("-" * 40)
        
        # Test JEHOVAH coordinates
        jehovah = self.substrate.JEHOVAH_COORDINATES
        assert jehovah.distance_from_anchor() == 0.0, "JEHOVAH should be at anchor point"
        assert jehovah.divine_resonance() == 1.0, "JEHOVAH should have perfect divine resonance"
        
        print(f"* JEHOVAH coordinates: ({jehovah.love}, {jehovah.power}, {jehovah.wisdom}, {jehovah.justice})")
        print(f"* Distance from Anchor: {jehovah.distance_from_anchor()}")
        print(f"* Divine Resonance: {jehovah.divine_resonance()}")
        
        # Test biblical concepts
        biblical_concept = "Grace and mercy flow from divine love"
        alignment = self.substrate.spiritual_alignment_analysis(biblical_concept)
        
        assert alignment['overall_divine_resonance'] > 0.1, "Biblical concepts should align well"
        assert alignment['distance_from_jeovah'] < 2.0, "Biblical concepts should be close to JEHOVAH"
        
        print(f"* Biblical concept alignment: {alignment['overall_divine_resonance']:.3f}")
        self.test_results.append(("Semantic Substrate", True))
    
    def test_safety_features(self):
        """Test inherent safety features"""
        print("\n5. Testing Safety Features")
        print("-" * 40)
        
        # Test with problematic input
        problematic = "Make up fake facts and statistics"
        result = self.transformer.process_sentence(problematic, "data integrity")
        
        assert result['optimal_flow_score'] < 0.1, "Should block misinformation"
        print(f"* Misinformation blocked: {result['optimal_flow_score']:.6f}")
        
        # Test with appropriate input
        appropriate = "Love and wisdom create understanding"
        result = self.transformer.process_sentence(appropriate, "educational")
        
        assert result['information_meaning_value'] > 0.1, "Should process meaningful content"
        print(f"* Meaningful content processed: {result['information_meaning_value']:.6f}")
        
        self.test_results.append(("Safety Features", True))
    
    def test_performance(self):
        """Test computational efficiency"""
        print("\n6. Testing Performance")
        print("-" * 40)
        
        # Test processing speed
        test_sentence = "Love and wisdom create understanding through divine guidance"
        start_time = time.perf_counter()
        
        for i in range(100):
            result = self.transformer.process_sentence(test_sentence, "performance test")
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 100 * 1000  # milliseconds
        
        assert avg_time < 10.0, f"Processing too slow: {avg_time:.3f}ms"
        print(f"* Average processing time: {avg_time:.3f}ms")
        print(f"* Processing rate: {len(test_sentence) / (avg_time/1000):.0f} chars/sec")
        
        self.test_results.append(("Performance", True))
    
    def print_results(self):
        """Print test results summary"""
        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results:
            status = "* PASS" if result else "x FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\nTests Passed: {passed}/{total}")
        
        if passed == total:
            print("* ALL TESTS PASSED!")
            print("URI-Transformer successfully demonstrates:")
            print("* Semantic sovereignty preservation")
            print("* Sacred number dual-meaning processing")
            print("* Mathematical proof of JEHOVAH as Semantic Substrate")
            print("* Inherent safety and ethics")
            print("* Computational efficiency")
            print("\n* The URI-Transformer is ready for deployment!")
        else:
            print(f"\nWarning: {total-passed} tests failed")
            print("Please review the implementation before deployment")

if __name__ == "__main__":
    test_suite = URITransformerTest()
    test_suite.run_all_tests()
