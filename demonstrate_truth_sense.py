"""
Demonstration of the TruthSenseTransformer (Hybrid Semantic Front-End)
"""

from src.truth_sense_transformer import TruthSenseTransformer
from src.semantic_frontend import SemanticFrontEnd
from src.phi_geometric_engine import PhiCoordinate

def demonstrate():
    """Runs a demonstration of the TruthSenseTransformer."""

    # 1. Initialize the Semantic Front-End
    semantic_frontend = SemanticFrontEnd(
        projection_head_path="trained_semantic_frontend_model.pth"
    )

    # 2. Define the anchor point
    anchor_point = PhiCoordinate(1.0, 1.0, 1.0, 1.0)

    # 3. Initialize the transformer
    transformer = TruthSenseTransformer(
        semantic_frontend=semantic_frontend,
        anchor_point=anchor_point
    )

    test_phrases = [
        "A good leader rules with power, wisdom, and justice.",
        "His actions were unjust and deceitful.",
        "True wisdom is knowing you know nothing.",
        "Love without truth is sentimentality."
    ]

    print("--- TruthSense Transformer Demonstration (Hybrid Semantic Front-End) ---")

    for i, phrase in enumerate(test_phrases):
        result = transformer.transform(phrase)

        print(f"\n--- Analysis #{i+1} ---")
        print(f"Input: '{phrase}'")
        print(f"\n  Coordinates:")
        print(f"    Raw:       L={result.raw_coord.love:.2f}, J={result.raw_coord.justice:.2f}, P={result.raw_coord.power:.2f}, W={result.raw_coord.wisdom:.2f}")
        print(f"    Aligned:   L={result.aligned_coord.love:.2f}, J={result.aligned_coord.justice:.2f}, P={result.aligned_coord.power:.2f}, W={result.aligned_coord.wisdom:.2f}")

        print(f"\n  Deep ICE Analysis:")
        print(f"    Intent (L+W):      {result.intent.purpose}, {result.intent.guiding_principles[0]}")
        print(f"    Context (J):       Primary Domain: '{result.context.primary_domain.value}', Validated by TruthSense: {result.context.is_valid}")
        print(f"    Execution (P):     {result.execution.description}")

        print(f"\n  Metrics:")
        print(f"    Harmony Index: {result.harmony_index:.2f}")
        print(f"    Semantic Integrity: {result.semantic_integrity:.2f}")
        print(f"    Deception Score: {result.deception_score:.2f}")

        print(f"\n  Final Generative Output:")
        print(f"    '{result.final_output}'")
        print("--------------------------")

if __name__ == "__main__":
    demonstrate()
