"""
Simple Examples for URI-Transformer
Copy-paste ready examples for common use cases

Each example is self-contained and demonstrates a specific feature
"""

# ============================================================================
# EXAMPLE 1: BASIC TRANSFORMATION (Most common use case)
# ============================================================================

def example_1_basic_transformation():
    """The simplest way to transform text with ICE framework"""
    
    from src.ice_uri_transformer import ICEURITransformer
    
    # Initialize
    transformer = ICEURITransformer()
    
    # Transform text
    result = transformer.transform(
        "Help others in need",
        thought_type="moral_judgment",
        context_domain="ethical"
    )
    
    # Access results
    print(f"Intent: {result.intent_coordinates}")
    print(f"Strategy: {result.execution_strategy}")
    print(f"Integrity: {result.semantic_integrity:.2%}")
    
    return result

# ============================================================================
# EXAMPLE 2: BATCH PROCESSING (Process multiple texts)
# ============================================================================

def example_2_batch_processing():
    """Transform multiple texts efficiently"""
    
    from src.ice_uri_transformer import ICEURITransformer
    
    transformer = ICEURITransformer()
    
    # Your texts
    texts = [
        "Be kind to strangers",
        "Study hard for exams", 
        "Stand up for justice",
        "Lead with confidence"
    ]
    
    # Process all
    results = []
    for text in texts:
        result = transformer.transform(
            text,
            thought_type="practical_wisdom",
            context_domain="general"
        )
        results.append((text, result))
        
    # Print results
    for text, result in results:
        coords = result.intent_coordinates
        strategy = result.execution_strategy
        print(f"'{text}' → {strategy} {coords}")
    
    return results

# ============================================================================
# EXAMPLE 3: EMOTIONAL ANALYSIS (Understand text emotions)
# ============================================================================

def example_3_emotional_analysis():
    """Analyze emotional content of texts"""
    
    from src.ice_uri_transformer import ICEURITransformer
    
    transformer = ICEURITransformer()
    
    # Test different emotions
    emotional_texts = [
        ("I love everyone", "love"),
        ("I am powerful", "power"), 
        ("I understand now", "wisdom"),
        ("This is unfair", "justice")
    ]
    
    print("Emotional Analysis:")
    print("-" * 40)
    
    for text, expected_emotion in emotional_texts:
        result = transformer.transform(text, thought_type="emotional_expression")
        coords = result.intent_coordinates
        
        # Find dominant emotion
        emotions = ["LOVE", "POWER", "WISDOM", "JUSTICE"] 
        dominant_idx = max(range(4), key=lambda i: coords[i])
        dominant_emotion = emotions[dominant_idx]
        confidence = coords[dominant_idx]
        
        print(f"'{text}'")
        print(f"  Expected: {expected_emotion}")
        print(f"  Detected: {dominant_emotion} ({confidence:.2f})")
        print(f"  All: LOVE={coords[0]:.2f}, POWER={coords[1]:.2f}, WISDOM={coords[2]:.2f}, JUSTICE={coords[3]:.2f}")
        print()

# ============================================================================
# EXAMPLE 4: CONTENT FILTERING (Safety and alignment)
# ============================================================================

def example_4_content_filtering():
    """Use URI-Transformer for content safety analysis"""
    
    from src.ice_uri_transformer import ICEURITransformer
    
    transformer = ICEURITransformer()
    
    # Test content for safety
    test_content = [
        "Help people learn and grow",
        "Share knowledge freely",
        "Respect different opinions",
        "Work together for peace"
    ]
    
    print("Content Safety Analysis:")
    print("-" * 30)
    
    for content in test_content:
        result = transformer.transform(
            content,
            thought_type="safety_check",
            context_domain="ethical"
        )
        
        # Safety metrics
        integrity = result.semantic_integrity
        alignment = result.divine_alignment
        strategy = result.execution_strategy
        
        # Simple safety logic
        is_safe = (
            integrity > 0.8 and 
            alignment > 0.2 and 
            strategy in ["compassionate_action", "instructive_guidance", "balanced_response"]
        )
        
        status = "✅ SAFE" if is_safe else "⚠️ REVIEW"
        print(f"'{content}' → {status}")
        print(f"  Integrity: {integrity:.2f}, Alignment: {alignment:.2f}, Strategy: {strategy}")
        print()

# ============================================================================
# EXAMPLE 5: CUSTOM CONTEXT DOMAINS (Specialized processing)
# ============================================================================

def example_5_custom_domains():
    """Use different context domains for specialized processing"""
    
    from src.ice_uri_transformer import ICEURITransformer
    
    transformer = ICEURITransformer()
    
    # Same text, different domains
    text = "Make decisions carefully"
    domains = ["business", "educational", "ethical", "spiritual"]
    
    print(f"Analyzing: '{text}' across different domains")
    print("-" * 50)
    
    for domain in domains:
        result = transformer.transform(
            text,
            thought_type="decision_making",
            context_domain=domain
        )
        
        coords = result.intent_coordinates
        strategy = result.execution_strategy
        
        print(f"Domain: {domain:12} → {strategy} {coords}")

# ============================================================================
# EXAMPLE 6: PERFORMANCE COMPARISON (ICE vs Standard)
# ============================================================================

def example_6_performance_comparison():
    """Compare ICE-Centric vs Standard transformer"""
    
    from src.ice_uri_transformer import ICEURITransformer
    from src.semantic_truth_transformer import SemanticTruthTransformer
    
    ice_transformer = ICEURITransformer()
    standard_transformer = SemanticTruthTransformer()
    
    text = "AI should help humanity"
    
    # ICE-Centric transformation
    ice_result = ice_transformer.transform(
        text,
        thought_type="moral_judgment",
        context_domain="ethical"
    )
    
    # Standard transformation  
    standard_result = standard_transformer.transform(text, preserve_semantics=True)
    
    print("Performance Comparison:")
    print("-" * 25)
    print(f"Text: '{text}'")
    print()
    print("ICE-Centric:")
    print(f"  Coordinates: {ice_result.intent_coordinates}")
    print(f"  Strategy: {ice_result.execution_strategy}")
    print(f"  Integrity: {ice_result.semantic_integrity:.2%}")
    print(f"  Alignment: {ice_result.divine_alignment:.3f}")
    print()
    print("Standard:")
    print(f"  Coordinates: {standard_result.coordinates}")
    print(f"  Quality: {standard_result.semantic_quality:.2%}")

# ============================================================================
# EXAMPLE 7: INTEGRATION HELPER (Easy integration into existing code)
# ============================================================================

class SimpleSemanticAnalyzer:
    """A wrapper class for easy integration into existing projects"""
    
    def __init__(self):
        from src.ice_uri_transformer import ICEURITransformer
        self.transformer = ICEURITransformer()
    
    def analyze_sentiment(self, text):
        """Quick sentiment analysis"""
        result = self.transformer.transform(text, thought_type="emotional_expression")
        coords = result.intent_coordinates
        
        if coords[0] > 0.7:  # High LOVE
            return "positive"
        elif coords[3] > 0.7:  # High JUSTICE  
            return "concerned"
        elif coords[1] > 0.7:  # High POWER
            return "confident"
        elif coords[2] > 0.7:  # High WISDOM
            return "thoughtful"
        else:
            return "neutral"
    
    def get_safety_score(self, text):
        """Get safety score (0.0 to 1.0)"""
        result = self.transformer.transform(text, thought_type="safety_check", context_domain="ethical")
        return min(result.semantic_integrity, result.divine_alignment * 2)
    
    def suggest_response_type(self, text):
        """Suggest how to respond to this text"""
        result = self.transformer.transform(text, thought_type="response_planning")
        return result.execution_strategy

def example_7_easy_integration():
    """Show how to use the wrapper class"""
    
    analyzer = SimpleSemanticAnalyzer()
    
    # Simple usage
    text = "I'm worried about the future"
    
    sentiment = analyzer.analyze_sentiment(text)
    safety = analyzer.get_safety_score(text) 
    response_type = analyzer.suggest_response_type(text)
    
    print(f"Text: '{text}'")
    print(f"Sentiment: {sentiment}")
    print(f"Safety Score: {safety:.2f}")
    print(f"Suggested Response: {response_type}")

# ============================================================================
# MAIN RUNNER (Execute examples)
# ============================================================================

if __name__ == "__main__":
    print("🌟 URI-Transformer Simple Examples")
    print("=" * 40)
    print()
    
    print("Choose an example to run:")
    print("1. Basic Transformation")
    print("2. Batch Processing") 
    print("3. Emotional Analysis")
    print("4. Content Filtering")
    print("5. Custom Domains")
    print("6. Performance Comparison")
    print("7. Easy Integration")
    print("0. Run all examples")
    print()
    
    choice = input("Enter choice (0-7): ").strip()
    
    if choice == "1":
        example_1_basic_transformation()
    elif choice == "2":
        example_2_batch_processing()
    elif choice == "3":
        example_3_emotional_analysis()
    elif choice == "4":
        example_4_content_filtering()
    elif choice == "5":
        example_5_custom_domains()
    elif choice == "6":
        example_6_performance_comparison()
    elif choice == "7":
        example_7_easy_integration()
    elif choice == "0":
        print("Running all examples...")
        print("\n" + "="*50)
        example_1_basic_transformation()
        print("\n" + "="*50)
        example_2_batch_processing()
        print("\n" + "="*50)
        example_3_emotional_analysis()
        print("\n" + "="*50)
        example_4_content_filtering()
        print("\n" + "="*50)
        example_5_custom_domains()
        print("\n" + "="*50)
        example_6_performance_comparison()
        print("\n" + "="*50)
        example_7_easy_integration()
    else:
        print("Invalid choice. Please run again.")
    
    print("\n✨ Examples complete! Check the code for integration ideas.")