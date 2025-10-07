#!/usr/bin/env python3
"""
Quick test for Guardian Cybersecurity Engine
"""

try:
    import sys
    import os
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    from guardian_engine import GuardianEngine
    import numpy as np
    
    print("✓ Dependencies loaded successfully")
    
    # Test the engine
    engine = GuardianEngine()
    
    # Test analysis
    test_concept = "malicious network attack"
    result = engine.analyze_threat(test_concept)
    
    print(f"✓ Engine test successful")
    print(f"Concept: {test_concept}")
    print(f"Threat Level: {result['threat_level']}")
    print(f"Coordinates: {result['sematrix']}")
    
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install numpy scipy scikit-learn pandas requests")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)