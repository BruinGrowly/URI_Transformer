"""
Demonstration of the TruthSenseTransformer (Deep & Generative ICE)
===================================================================

This script showcases the full capabilities of the refactored TruthSenseTransformer,
demonstrating its deep, integrated ICE pipeline and its new generative output.
"""

from src.truth_sense_transformer import TruthSenseTransformer

def demonstrate():
    """Runs a demonstration of the TruthSenseTransformer."""

    transformer = TruthSenseTransformer()

    test_phrases = [
        "Love is the foundation of a just and powerful society, guided by wisdom.",
        "Hate and division lead to the collapse of civilizations.", # Should have low Justice
        "A truly powerful leader serves with humility and compassion."
    ]

    print("--- TruthSense Transformer Demonstration (Deep & Generative ICE) ---")

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

        print(f"\n  Final Generative Output:")
        print(f"    '{result.final_output}'")
        print("--------------------------")

if __name__ == "__main__":
    demonstrate()
