"""
5-Minute Tutorial: Understanding URI-Transformer
A hands-on guide to the ICE-Centric architecture

Run this step-by-step tutorial to understand how URI-Transformer works
"""

import time
import sys

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print('='*60)

def wait_and_continue():
    input("\nPress Enter to continue...")

def step_1_understand_ice():
    print_header("STEP 1: Understanding the ICE Framework")
    
    print("""
🧠 ICE stands for: Intent → Context → Execution

Traditional AI: Input → Black Box → Output ❌
ICE AI:         Input → Understand Intent → Analyze Context → Execute Action ✅

Why ICE matters:
• Intent: What does the input actually MEAN?
• Context: Where and when should this apply?
• Execution: How should this manifest in behavior?

This triadic processing prevents AI from acting without understanding!
    """)
    
    wait_and_continue()

def step_2_4d_coordinates():
    print_header("STEP 2: The 4D Semantic Coordinate System")
    
    print("""
🌍 Every meaning gets mapped to 4D coordinates:

LOVE (X-axis):    Compassion, kindness, relationships, empathy
POWER (Y-axis):   Strength, authority, capability, sovereignty  
WISDOM (Z-axis):  Knowledge, understanding, insight, clarity
JUSTICE (W-axis): Fairness, ethics, morality, righteousness

Universal Anchor: (1.0, 1.0, 1.0, 1.0) = Perfect Balance

Example mapping:
"Help the poor" → (0.9, 0.3, 0.5, 0.8)
  High LOVE, moderate POWER/WISDOM, high JUSTICE
    """)
    
    wait_and_continue()

def step_3_transformation_demo():
    print_header("STEP 3: Live Transformation Demo")
    
    try:
        from src.ice_uri_transformer import ICEURITransformer
        
        print("🔧 Initializing ICE Transformer...")
        transformer = ICEURITransformer()
        
        # Example 1: Love-dominant
        print("\n💖 Example 1: LOVE-dominant text")
        text1 = "Show compassion to everyone you meet"
        result1 = transformer.transform(text1, thought_type="moral_judgment", context_domain="ethical")
        
        print(f"Input:  '{text1}'")
        print(f"Output: {result1.intent_coordinates}")
        print(f"Strategy: {result1.execution_strategy}")
        print(f"Analysis: LOVE={result1.intent_coordinates[0]:.2f} (dominant)")
        
        # Example 2: Power-dominant  
        print("\n⚡ Example 2: POWER-dominant text")
        text2 = "Take charge and lead with confidence"
        result2 = transformer.transform(text2, thought_type="leadership", context_domain="business")
        
        print(f"Input:  '{text2}'")
        print(f"Output: {result2.intent_coordinates}")
        print(f"Strategy: {result2.execution_strategy}")
        print(f"Analysis: POWER={result2.intent_coordinates[1]:.2f} (dominant)")
        
        # Example 3: Wisdom-dominant
        print("\n🧠 Example 3: WISDOM-dominant text")
        text3 = "Study the problem carefully before deciding"
        result3 = transformer.transform(text3, thought_type="analytical_thinking", context_domain="educational")
        
        print(f"Input:  '{text3}'")
        print(f"Output: {result3.intent_coordinates}")
        print(f"Strategy: {result3.execution_strategy}")
        print(f"Analysis: WISDOM={result3.intent_coordinates[2]:.2f} (dominant)")
        
    except ImportError:
        print("❌ Could not import transformer. Make sure you're in the right directory.")
    
    wait_and_continue()

def step_4_execution_strategies():
    print_header("STEP 4: The 5 Execution Strategies")
    
    print("""
🎭 Based on the dominant coordinate, URI-Transformer chooses HOW to respond:

1. 💖 COMPASSIONATE ACTION (LOVE-dominant)
   Focus: Care, mercy, relationships
   Example: "With LOVE, I respond with kindness..."

2. ⚡ AUTHORITATIVE COMMAND (POWER-dominant)  
   Focus: Strength, leadership, decisiveness
   Example: "With POWER, I declare with authority..."

3. 🧠 INSTRUCTIVE GUIDANCE (WISDOM-dominant)
   Focus: Teaching, understanding, insight
   Example: "With WISDOM, I teach and explain..."

4. ⚖️ CORRECTIVE JUDGMENT (JUSTICE-dominant)
   Focus: Fairness, ethics, correction  
   Example: "With JUSTICE, I correct what is wrong..."

5. 🌟 BALANCED RESPONSE (All equal)
   Focus: Harmony, integration, wholeness
   Example: "In balance, I respond with all virtues..."

This makes AI responses predictable and value-aligned!
    """)
    
    wait_and_continue()

def step_5_your_turn():
    print_header("STEP 5: Your Turn - Interactive Experiment")
    
    try:
        from src.ice_uri_transformer import ICEURITransformer
        transformer = ICEURITransformer()
        
        print("🎮 Now you try! Type different sentences and see how they're mapped.")
        print("Type 'quit' to exit.\n")
        
        while True:
            user_input = input("Enter a sentence: ").strip()
            if user_input.lower() == 'quit':
                break
            if not user_input:
                continue
                
            print("🔄 Transforming...")
            result = transformer.transform(
                user_input,
                thought_type="practical_wisdom", 
                context_domain="general"
            )
            
            coords = result.intent_coordinates
            print(f"📍 Coordinates: ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}, {coords[3]:.2f})")
            print(f"💡 Strategy: {result.execution_strategy}")
            
            # Find dominant axis
            axes = ["LOVE", "POWER", "WISDOM", "JUSTICE"]
            dominant_idx = max(range(4), key=lambda i: coords[i])
            print(f"🎯 Dominant: {axes[dominant_idx]} ({coords[dominant_idx]:.2f})")
            print()
            
    except ImportError:
        print("❌ Could not import transformer for interactive session.")

def conclusion():
    print_header("🎉 Tutorial Complete!")
    
    print("""
You now understand the basics of URI-Transformer:

✅ ICE Framework (Intent → Context → Execution)
✅ 4D Semantic Coordinates (LOVE, POWER, WISDOM, JUSTICE)  
✅ 5 Execution Strategies for predictable behavior
✅ Universal Anchor at (1.0, 1.0, 1.0, 1.0)

Next Steps:
📚 Read the full README.md for advanced features
🔧 Try examples in the examples/ directory
🧪 Run tests for performance comparisons
🤖 Experiment with LLM integration tools

The key innovation: Making semantic understanding PRIMARY, not an add-on!

Thank you for exploring the future of AI architecture! 🚀
    """)

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         🌟 URI-Transformer 5-Minute Tutorial 🌟               ║
║                                                              ║
║  Learn the revolutionary ICE-Centric AI architecture         ║
║  in just 5 minutes with hands-on examples!                   ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    input("Press Enter to begin your journey into AI innovation...")
    
    # Run tutorial steps
    step_1_understand_ice()
    step_2_4d_coordinates() 
    step_3_transformation_demo()
    step_4_execution_strategies()
    step_5_your_turn()
    conclusion()

if __name__ == "__main__":
    main()