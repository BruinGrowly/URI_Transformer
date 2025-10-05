#!/usr/bin/env python3
"""
Quick test to verify the URI-Transformer installation
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from uri_transformer import URITransformer
    print("SUCCESS: URI-Transformer imports successfully")
    
    from semantic_substrate import SemanticSubstrate
    print("SUCCESS: Semantic Substrate imports successfully")
    
    # Test basic functionality
    transformer = URITransformer()
    substrate = SemanticSubstrate()
    
    # Test semantic unit creation
    unit = transformer.create_semantic_unit("love", "divine context")
    print(f"SUCCESS: Semantic unit created: {unit.word}")
    
    # Test bridge function
    result = transformer.process_sentence("Love creates understanding", "educational")
    print(f"SUCCESS: Sentence processed: {result['information_meaning_value']:.6f}")
    
    # Test semantic substrate
    jehovah = substrate.JEHOVAH_COORDINATES
    print(f"SUCCESS: JEHOVAH coordinates: ({jehovah.love}, {jehovah.power}, {jehovah.wisdom}, {jehovah.justice})")
    
    print("\nALL TESTS PASSED: URI-Transformer is ready for use!")
    
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
except Exception as e:
    print(f"ERROR: Test failed: {e}")
