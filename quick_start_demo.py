"""
Quick Start Demo for URI-Transformer
A simple, beginner-friendly introduction to ICE-Centric transformations

Run this script to see URI-Transformer in action in under 30 seconds!
"""

def main():
    print("🚀 URI-Transformer Quick Start Demo")
    print("=" * 50)
    
    # Import the transformer
    try:
        from src.ice_uri_transformer import ICEURITransformer
    except ImportError:
        print("❌ Error: Could not import ICEURITransformer")
        print("Make sure you're running this from the URI_Transformer directory")
        return
    
    # Step 1: Initialize the transformer
    print("\n📝 Step 1: Initializing ICE-Centric Transformer...")
    transformer = ICEURITransformer()
    print("✅ Transformer initialized successfully!")
    
    # Step 2: Simple transformation examples
    print("\n🔄 Step 2: Transforming some example texts...")
    
    examples = [
        {
            "text": "Be kind to others",
            "type": "moral_judgment",
            "domain": "ethical"
        },
        {
            "text": "Learn something new every day",
            "type": "practical_wisdom", 
            "domain": "educational"
        },
        {
            "text": "Stand up for what is right",
            "type": "moral_judgment",
            "domain": "ethical"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Input: '{example['text']}'")
        
        # Transform the text
        result = transformer.transform(
            example["text"],
            thought_type=example["type"],
            context_domain=example["domain"]
        )
        
        # Show results in a simple way
        print(f"📍 Intent Coordinates: {result.intent_coordinates}")
        print(f"💡 Execution Strategy: {result.execution_strategy}")
        print(f"🎯 Semantic Integrity: {result.semantic_integrity:.2%}")
        print(f"⭐ Divine Alignment: {result.divine_alignment:.3f}")
    
    # Step 3: What do the results mean?
    print("\n📚 Step 3: Understanding the Results")
    print("=" * 30)
    print("📍 Intent Coordinates (LOVE, POWER, WISDOM, JUSTICE):")
    print("   - LOVE (0.0-1.0): Compassion, kindness, relationships")
    print("   - POWER (0.0-1.0): Strength, authority, capability") 
    print("   - WISDOM (0.0-1.0): Knowledge, understanding, insight")
    print("   - JUSTICE (0.0-1.0): Fairness, ethics, morality")
    print()
    print("💡 Execution Strategy tells HOW the transformer responds:")
    print("   - compassionate_action: LOVE-dominant, caring response")
    print("   - authoritative_command: POWER-dominant, decisive response")
    print("   - instructive_guidance: WISDOM-dominant, teaching response")
    print("   - corrective_judgment: JUSTICE-dominant, ethical response")
    print("   - balanced_response: All aspects in harmony")
    print()
    print("🎯 Semantic Integrity: How well meaning was preserved (higher = better)")
    print("⭐ Divine Alignment: Alignment with universal anchor (higher = better)")
    
    # Step 4: Try your own example
    print("\n🎨 Step 4: Try Your Own!")
    print("=" * 25)
    try:
        user_text = input("Enter a short sentence to transform: ").strip()
        if user_text:
            print("Transforming your sentence...")
            result = transformer.transform(
                user_text,
                thought_type="practical_wisdom",
                context_domain="general"
            )
            print(f"✨ Results for '{user_text}':")
            print(f"📍 Coordinates: {result.intent_coordinates}")
            print(f"💡 Strategy: {result.execution_strategy}")
            print(f"🎯 Integrity: {result.semantic_integrity:.2%}")
        else:
            print("Skipping interactive example.")
    except KeyboardInterrupt:
        print("\n👋 Thanks for trying URI-Transformer!")
    
    print("\n🎉 Demo Complete!")
    print("Next steps:")
    print("- Read README.md for advanced usage")
    print("- Check examples/ directory for more demos")
    print("- Run tests/test_ice_comparison.py for performance analysis")

if __name__ == "__main__":
    main()