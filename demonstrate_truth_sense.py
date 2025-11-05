"""
Demonstration of the TruthSenseTransformer
"""

from src.truth_sense_transformer import TruthSenseTransformer
from src.semantic_frontend import SemanticFrontEnd
from src.phi_geometric_engine import PhiCoordinate, universal_semantic_mix, quadratic_semantic_mix, golden_semantic_mix
import numpy as np

def demonstrate_text_analysis(transformer: TruthSenseTransformer):
    """Runs a demonstration of the TruthSenseTransformer with text input."""
    test_phrases = [
        "A good leader rules with power, wisdom, and justice.",
        "His actions were unjust and deceitful.",
        "True wisdom is knowing you know nothing.",
        "Love without truth is sentimentality.",
        "Effective leadership requires empowering and serving others."
    ]

    print("--- TruthSense Transformer Text Analysis Demonstration ---")

    for i, phrase in enumerate(test_phrases):
        result = transformer.transform(phrase)

        print(f"\n--- Analysis #{i+1} ---")
        print(f"Input: '{phrase}'\n")
        print(result.final_output)
        print("\n--------------------------")

def demonstrate_palette(transformer: TruthSenseTransformer):
    """Runs a demonstration of the Semantic Color Palette Interface."""
    print("\n--- Semantic Color Palette Interface ---")
    print("Design new concepts by mixing Love, Justice, Power, and Wisdom.")
    print("Enter weights (e.g., 1.0, 0.5, 2.0, 1.0) or 'q' to quit.")

    while True:
        try:
            print("\nEnter weights for (Love, Justice, Power, Wisdom) separated by spaces:")
            weights_input = input("> ")
            if weights_input.lower() == 'q':
                break

            weights_list = list(map(float, weights_input.split()))
            if len(weights_list) != 4:
                print("Please enter exactly four weights.")
                continue

            primary_weights = {
                'love': weights_list[0],
                'justice': weights_list[1],
                'power': weights_list[2],
                'wisdom': weights_list[3]
            }

            print("Select mixing method (linear, quadratic, golden) [default: linear]:")
            method_input = input("> ").lower()
            mixing_method = method_input if method_input in ["linear", "quadratic", "golden"] else "linear"

            result = transformer.generate_from_recipe(primary_weights, mixing_method=mixing_method)

            print(f"\n--- Generated Concept ({mixing_method.capitalize()} Mixing) ---")
            print(f"Input Weights: {primary_weights}")
            print(f"Raw PhiCoordinate: L={result.raw_coord.love:.3f}, J={result.raw_coord.justice:.3f}, P={result.raw_coord.power:.3f}, W={result.raw_coord.wisdom:.3f}")
            print(f"Aligned PhiCoordinate: L={result.aligned_coord.love:.3f}, J={result.aligned_coord.justice:.3f}, P={result.aligned_coord.power:.3f}, W={result.aligned_coord.wisdom:.3f}")
            print(result.final_output)
            print("\n--------------------------")

        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces, or 'q'.")
        except Exception as e:
            print(f"An error occurred: {e}")

def demonstrate_code_analysis(transformer: TruthSenseTransformer):
    """Runs a demonstration of the TruthSenseTransformer with code input."""
    print("\n--- Code Analysis Interface ---")
    print("Enter Python code snippets (type 'END' on a new line to finish, or 'q' to quit).")

    while True:
        print("\nEnter your code snippet:")
        code_lines = []
        while True:
            line = input("> ")
            if line.lower() == 'end':
                break
            if line.lower() == 'q':
                return # Exit function
            code_lines.append(line)
        
        code_snippet = "\n".join(code_lines)
        if not code_snippet.strip():
            print("No code entered. Please try again.")
            continue

        try:
            result = transformer.transform_code(code_snippet)

            print(f"\n--- Code Semantic Analysis ---")
            print(f"Input Code:\n---\n{code_snippet}\n---")
            print(f"Raw PhiCoordinate: L={result.raw_coord.love:.3f}, J={result.raw_coord.justice:.3f}, P={result.raw_coord.power:.3f}, W={result.raw_coord.wisdom:.3f}")
            print(f"Aligned PhiCoordinate: L={result.aligned_coord.love:.3f}, J={result.aligned_coord.justice:.3f}, P={result.aligned_coord.power:.3f}, W={result.aligned_coord.wisdom:.3f}")
            print(result.final_output)
            print("\n--------------------------")

        except Exception as e:
            print(f"An error occurred during code analysis: {e}")


if __name__ == "__main__":
    # 1. Initialize the Semantic Front-End
    semantic_frontend = SemanticFrontEnd(
        projection_head_path="trained_semantic_frontend_model.pth"
    )

    # 2. Define the anchor point
    anchor_point = PhiCoordinate(1.0, 1.0, 1.0, 1.0)

    # 3. Initialize the transformer (default to linear mixing)
    transformer = TruthSenseTransformer(
        semantic_frontend=semantic_frontend,
        anchor_point=anchor_point,
        mixing_method="linear"
    )

    print("\nChoose a demonstration mode:")
    print("1. Text Analysis")
    print("2. Semantic Color Palette Interface")
    print("3. Code Analysis (default)")
    mode_choice = input("> ").strip()

    if mode_choice == "1":
        demonstrate_text_analysis(transformer)
    elif mode_choice == "2":
        demonstrate_palette(transformer)
    else:
        demonstrate_code_analysis(transformer)
