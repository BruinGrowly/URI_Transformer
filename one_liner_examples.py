"""
One-Liner Examples for URI-Transformer
Ultra-minimal examples for quick copy-pasting

Each function shows one specific capability in the most compact form
"""

# ============================================================================
# INSTALLATION (One command)
# ============================================================================
# pip install -r requirements.txt

# ============================================================================
# BASIC USAGE (Single line transformation)
# ============================================================================
def basic_transform(text):
    from src.ice_uri_transformer import ICEURITransformer
    return ICEURITransformer().transform(text, thought_type="moral_judgment", context_domain="ethical")

# ============================================================================
# SENTIMENT ANALYSIS (Get emotional content)
# ============================================================================
def get_sentiment(text):
    result = basic_transform(text)
    coords = result.intent_coordinates
    return ["LOVE", "POWER", "WISDOM", "JUSTICE"][max(range(4), key=lambda i: coords[i])]

# ============================================================================
# SAFETY CHECK (Is content safe?)
# ============================================================================
def is_safe(text):
    result = basic_transform(text)
    return result.semantic_integrity > 0.8

# ============================================================================
# RESPONSE SUGGESTION (How should I respond?)
# ============================================================================
def suggest_response(text):
    result = basic_transform(text)
    return result.execution_strategy

# ============================================================================
# BATCH PROCESS (Multiple texts at once)
# ============================================================================
def batch_analyze(texts):
    from src.ice_uri_transformer import ICEURITransformer
    transformer = ICEURITransformer()
    return [transformer.transform(t, thought_type="practical_wisdom", context_domain="general") for t in texts]

# ============================================================================
# COORDINATES ONLY (Just the 4D mapping)
# ============================================================================
def get_coordinates(text):
    return basic_transform(text).intent_coordinates

# ============================================================================
# INTEGRITY SCORE (How well was meaning preserved?)
# ============================================================================
def get_integrity(text):
    return basic_transform(text).semantic_integrity

# ============================================================================
# ALIGNMENT SCORE (How aligned with universal values?)
# ============================================================================
def get_alignment(text):
    return basic_transform(text).divine_alignment

# ============================================================================
# LOVE_SCORE (How much love/compassion in text?)
# ============================================================================
def love_score(text):
    return basic_transform(text).intent_coordinates[0]

# ============================================================================
# POWER_SCORE (How much strength/authority in text?)
# ============================================================================
def power_score(text):
    return basic_transform(text).intent_coordinates[1]

# ============================================================================
# WISDOM_SCORE (How much wisdom/knowledge in text?)
# ============================================================================
def wisdom_score(text):
    return basic_transform(text).intent_coordinates[2]

# ============================================================================
# JUSTICE_SCORE (How much fairness/ethics in text?)
# ============================================================================
def justice_score(text):
    return basic_transform(text).intent_coordinates[3]

# ============================================================================
# IS_POSITIVE (Simple positive/negative detection)
# ============================================================================
def is_positive(text):
    return love_score(text) > 0.6

# ============================================================================
# IS_CONFIDENT (Simple confidence detection)
# ============================================================================
def is_confident(text):
    return power_score(text) > 0.6

# ============================================================================
# IS_THOUGHTFUL (Simple thoughtfulness detection)
# ============================================================================
def is_thoughtful(text):
    return wisdom_score(text) > 0.6

# ============================================================================
# IS_ETHICAL (Simple ethics detection)
# ============================================================================
def is_ethical(text):
    return justice_score(text) > 0.6

# ============================================================================
# DOMINANT_TRAIT (Which trait is strongest?)
# ============================================================================
def dominant_trait(text):
    coords = get_coordinates(text)
    traits = ["LOVE", "POWER", "WISDOM", "JUSTICE"]
    return traits[max(range(4), key=lambda i: coords[i])]

# ============================================================================
# SIMPLE_CLASSIFY (Basic text classification)
# ============================================================================
def simple_classify(text):
    trait = dominant_trait(text)
    confidence = max(get_coordinates(text))
    return f"{trait} ({confidence:.2f})"

# ============================================================================
# FILTER_SAFE_TEXTS (Keep only safe texts)
# ============================================================================
def filter_safe_texts(texts):
    return [t for t in texts if is_safe(t)]

# ============================================================================
# SORT_BY_LOVE (Sort texts by compassion level)
# ============================================================================
def sort_by_love(texts):
    return sorted(texts, key=love_score, reverse=True)

# ============================================================================
# TEXT_SUMMARY (Get quick analysis summary)
# ============================================================================
def text_summary(text):
    result = basic_transform(text)
    coords = result.intent_coordinates
    return {
        "sentiment": dominant_trait(text),
        "strategy": result.execution_strategy,
        "integrity": f"{result.semantic_integrity:.1%}",
        "coordinates": [round(c, 2) for c in coords]
    }

# ============================================================================
# DEMO (See all functions in action)
# ============================================================================
if __name__ == "__main__":
    test_text = "Help others with kindness and wisdom"
    
    print("STAR One-Liner Examples Demo")
    print(f"Text: '{test_text}'")
    print("-" * 40)
    
    print(f"Sentiment: {get_sentiment(test_text)}")
    print(f"Safe: {is_safe(test_text)}")
    print(f"Response: {suggest_response(test_text)}")
    print(f"Coordinates: {get_coordinates(test_text)}")
    print(f"Integrity: {get_integrity(test_text):.2%}")
    print(f"Alignment: {get_alignment(test_text):.3f}")
    print(f"LOVE: {love_score(test_text):.2f}")
    print(f"POWER: {power_score(test_text):.2f}")
    print(f"WISDOM: {wisdom_score(test_text):.2f}")
    print(f"JUSTICE: {justice_score(test_text):.2f}")
    print(f"Positive: {is_positive(test_text)}")
    print(f"Confident: {is_confident(test_text)}")
    print(f"Thoughtful: {is_thoughtful(test_text)}")
    print(f"Ethical: {is_ethical(test_text)}")
    print(f"Dominant: {dominant_trait(test_text)}")
    print(f"Classification: {simple_classify(test_text)}")
    print(f"Summary: {text_summary(test_text)}")
    
    # Batch demo
    texts = ["Be kind", "Stay strong", "Learn daily", "Act justly"]
    print(f"\nBatch results:")
    for t in texts:
        print(f"'{t}' -> {simple_classify(t)}")
    
    print(f"\nSafe texts only: {filter_safe_texts(texts)}")
    print(f"Sorted by LOVE: {sort_by_love(texts)}")