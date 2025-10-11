"""
URI-Transformer - Universal Reality Interface

A revolutionary AI architecture that preserves semantic meaning while enabling
mathematical computation through a 4D coordinate system.

Main Components:
- SemanticTruthTransformer: Main transformer with semantic preservation
- SimpleTransformer: Lightweight implementation
- UltimateCoreEngine: Semantic analysis engine
- SemanticCalculus: Mathematical operations on meaning
"""

__version__ = "1.0.0"
__author__ = "BruinGrowly"
__license__ = "MIT"

# Core exports
try:
    from .semantic_truth_transformer import SemanticTruthTransformer
    from .simple_transformer import SimpleTransformer
    from .ultimate_core_engine import UltimateCoreEngine
    from .semantic_calculus import SemanticCalculus
    from .enhanced_core_components import EnhancedCoreComponents
    from .ice_framework import ICEFramework
    
    __all__ = [
        'SemanticTruthTransformer',
        'SimpleTransformer',
        'UltimateCoreEngine',
        'SemanticCalculus',
        'EnhancedCoreComponents',
        'ICEFramework',
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")
    __all__ = []
