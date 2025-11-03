"""
Demonstration of the TruthSenseTransformer
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
        "Love without truth is sentimentality.",
        "Effective leadership requires empowering and serving others."
    ]

    print("--- TruthSense Transformer Demonstration ---")

    for i, phrase in enumerate(test_phrases):
        result = transformer.transform(phrase)

        print(f"\n--- Analysis #{i+1} ---")
        print(f"Input: '{phrase}'\n")
        # The final output is now a multi-line, formatted string
        print(result.final_output)
        print("\n--------------------------")

if __name__ == "__main__":
    demonstrate()
