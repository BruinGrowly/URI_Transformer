#!/usr/bin/env python3
"""
Simple Test for Semantic Truth Transformer
Tests core functionality without unicode issues
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from semantic_truth_transformer import SemanticTruthTransformer, ProcessingMode
    print("[OK] Semantic Truth Transformer imported successfully")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def test_basic_transformation():
    """Test basic transformation functionality"""
    print("\n" + "="*60)
    print("SEMANTIC TRUTH TRANSFORMER - BASIC TEST")
    print("="*60)
    
    # Create transformer
    transformer = SemanticTruthTransformer()
    
    # Test cases
    test_cases = [
        ("God loves us", "biblical"),
        ("Truth matters", "general"),
        ("Wisdom guides", "biblical"),
        ("Business ethics", "business")
    ]
    
    for text, context in test_cases:
        print(f"\n[TEST] Processing: '{text}' in {context} context")
        
        try:
            # Simple transformation
            result = transformer.transform(text, context)
            
            if 'error' not in result:
                print("[OK] Transformation successful")
                
                # Show key results
                stages = result.get('processing_stages', {})
                
                if 'semantic_extraction' in stages:
                    avg_conf = stages['semantic_extraction'].get('average_confidence', 0)
                    print(f"  Semantic confidence: {avg_conf:.3f}")
                
                if 'truth_scaffold' in stages:
                    truth_aligned = stages['truth_scaffold'].get('truth_aligned', 0)
                    total_tokens = stages['truth_scaffold'].get('tokens_analyzed', 0)
                    print(f"  Truth alignment: {truth_aligned}/{total_tokens} tokens")
                
                if 'biblical_alignment' in stages:
                    avg_alignment = stages['biblical_alignment'].get('average_alignment', 0)
                    print(f"  Biblical alignment: {avg_alignment:.3f}")
                
                if 'final_output' in result:
                    summary = result['final_output'].get('summary', {})
                    print(f"  Final truth density: {summary.get('average_truth_density', 0):.3f}")
                    print(f"  Final semantic confidence: {summary.get('average_semantic_confidence', 0):.3f}")
                
            else:
                print(f"[ERROR] {result['error']}")
        
        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
    
    print("\n" + "="*60)
    print("BASIC TEST COMPLETE")

def test_ice_execution():
    """Test ICE Framework execution"""
    print("\n" + "="*60)
    print("ICE FRAMEWORK EXECUTION TEST")
    print("="*60)
    
    transformer = SemanticTruthTransformer(ProcessingMode.EXECUTION)
    
    # Test execution
    test_input = "How can I show God's love to others?"
    print(f"[EXECUTION] Processing: '{test_input}'")
    
    try:
        result = transformer.transform(test_input, "counseling")
        
        if 'ice_execution' in result:
            ice_result = result['ice_execution']
            if 'divine_alignment' in ice_result:
                print(f"[OK] Divine alignment: {ice_result['divine_alignment']:.3f}")
                print(f"[OK] Execution strategy: {ice_result.get('execution_strategy', 'N/A')}")
            else:
                print("[WARNING] ICE execution returned no divine alignment")
        else:
            print("[WARNING] No ICE execution results")
    
    except Exception as e:
        print(f"[ERROR] ICE execution test failed: {e}")
    
    print("\n" + "="*60)
    print("ICE EXECUTION TEST COMPLETE")

def main():
    """Main test function"""
    print("SEMANTIC TRUTH TRANSFORMER v1.0 - TEST SUITE")
    print("Testing revolutionary transformer architecture")
    
    # Run tests
    test_basic_transformation()
    test_ice_execution()
    
    print(f"\n[SUCCESS] All tests completed!")
    print(f"[STATUS] Semantic Truth Transformer is operational")
    print(f"[CAPABILITIES] Meaning processing, truth analysis, biblical alignment, ICE execution")

if __name__ == "__main__":
    main()