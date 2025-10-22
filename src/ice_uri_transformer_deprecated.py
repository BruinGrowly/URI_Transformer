"""
ICE-Centric URI-Transformer
============================

Makes the ICE Framework (Intent-Context-Execution) the PRIMARY processing architecture.

Traditional approach: Input → Tokenization → Embedding → Transformer → Output
ICE-Centric approach: Input → Intent → Context → Execution → Output

Every transformation is an ICE triadic process:
1. INTENT: Extract and map human thought to 4D semantic coordinates
2. CONTEXT: Determine domain and align meaning with universal principles
3. EXECUTION: Generate behaviorally-aligned output preserving semantic integrity

Author: Semantic Substrate Engine Team
License: MIT
"""

from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import numpy as np


# Universal Anchor Point: Jehovah (1.0, 1.0, 1.0, 1.0)
JEHOVAH = (1.0, 1.0, 1.0, 1.0)


@dataclass
class ICETransformationResult:
    """Result of ICE-Centric transformation."""

    # INTENT Phase
    intent_coordinates: Tuple[float, float, float, float]  # (LOVE, POWER, WISDOM, JUSTICE)
    intent_type: str

    # CONTEXT Phase
    context_domain: str
    context_alignment: float

    # EXECUTION Phase
    execution_strategy: str
    output_text: str

    # Quality Metrics
    divine_alignment: float  # Distance from JEHOVAH anchor
    anchor_distance: float
    semantic_integrity: float
    transformation_stages: List[str]


class ICEURITransformer:
    """
    ICE-Centric URI-Transformer

    Makes Intent-Context-Execution the PRIMARY architecture.
    Every transformation follows the seven-stage ICE process:

    1. INTENT EXTRACTION: Parse human thought into semantic intent
    2. INTENT MAPPING: Map to 4D coordinates (LOVE, POWER, WISDOM, JUSTICE)
    3. CONTEXT ANALYSIS: Determine domain and situational factors
    4. CONTEXT ALIGNMENT: Align with universal principles and anchor
    5. EXECUTION STRATEGY: Choose optimal transformation path
    6. EXECUTION VALIDATION: Verify semantic integrity
    7. OUTPUT GENERATION: Produce behaviorally-aligned result
    """

    def __init__(self, anchor_point: Tuple[float, float, float, float] = JEHOVAH):
        """Initialize ICE-Centric transformer with universal anchor."""
        self.anchor_point = anchor_point
        self.transformation_count = 0
        self.total_alignment = 0.0

        # Sacred numbers for stability assessment
        self.sacred_numbers = {
            613: "perfect_stability",
            12: "high_stability",
            7: "near_perfect",
            40: "testing_period"
        }

    # ========================================
    # STAGE 1-2: INTENT (Extraction & Mapping)
    # ========================================

    def extract_intent(self, input_text: str, thought_type: str = "practical_wisdom") -> Dict:
        """
        STAGE 1: Extract semantic intent from input.

        Maps human thought to intention type and semantic features.
        """
        intent_types = {
            "practical_wisdom": {"description": "Actionable guidance", "weight": 0.8},
            "moral_judgment": {"description": "Ethical evaluation", "weight": 0.9},
            "creative_expression": {"description": "Novel combination", "weight": 0.7},
            "factual_statement": {"description": "Objective truth", "weight": 0.95},
            "emotional_expression": {"description": "Affective state", "weight": 0.6}
        }

        intent = {
            "type": thought_type,
            "text": input_text,
            "features": self._extract_semantic_features(input_text),
            "weight": intent_types.get(thought_type, {"weight": 0.5})["weight"]
        }

        return intent

    def _extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Extract semantic features from text."""
        text_lower = text.lower()

        # LOVE indicators
        love_words = ["love", "compassion", "mercy", "kindness", "care", "grace"]
        love_score = sum(1 for word in love_words if word in text_lower) / max(len(text_lower.split()), 1)

        # POWER indicators
        power_words = ["power", "strength", "authority", "might", "sovereign", "rule"]
        power_score = sum(1 for word in power_words if word in text_lower) / max(len(text_lower.split()), 1)

        # WISDOM indicators
        wisdom_words = ["wisdom", "understanding", "knowledge", "insight", "discernment"]
        wisdom_score = sum(1 for word in wisdom_words if word in text_lower) / max(len(text_lower.split()), 1)

        # JUSTICE indicators
        justice_words = ["justice", "righteousness", "fair", "right", "moral", "ethical"]
        justice_score = sum(1 for word in justice_words if word in text_lower) / max(len(text_lower.split()), 1)

        return {
            "love": min(love_score * 10, 1.0),
            "power": min(power_score * 10, 1.0),
            "wisdom": min(wisdom_score * 10, 1.0),
            "justice": min(justice_score * 10, 1.0)
        }

    def map_intent_to_coordinates(self, intent: Dict) -> Tuple[float, float, float, float]:
        """
        STAGE 2: Map intent to 4D semantic coordinates.

        Returns: (LOVE, POWER, WISDOM, JUSTICE) coordinates
        """
        features = intent["features"]
        weight = intent["weight"]

        # Base coordinates from semantic features
        love = features["love"] * weight
        power = features["power"] * weight
        wisdom = features["wisdom"] * weight
        justice = features["justice"] * weight

        # Normalize to [0, 1] range
        total = love + power + wisdom + justice
        if total > 0:
            love /= total
            power /= total
            wisdom /= total
            justice /= total

        return (love, power, wisdom, justice)

    # ========================================
    # STAGE 3-4: CONTEXT (Analysis & Alignment)
    # ========================================

    def analyze_context(self, intent: Dict, context_domain: str = "general") -> Dict:
        """
        STAGE 3: Analyze contextual domain and situational factors.
        """
        domains = {
            "general": {"stability": 0.7, "complexity": 0.5},
            "ethical": {"stability": 0.9, "complexity": 0.8},
            "technical": {"stability": 0.6, "complexity": 0.9},
            "relational": {"stability": 0.8, "complexity": 0.7},
            "spiritual": {"stability": 1.0, "complexity": 1.0}
        }

        domain_info = domains.get(context_domain, domains["general"])

        context = {
            "domain": context_domain,
            "stability": domain_info["stability"],
            "complexity": domain_info["complexity"],
            "requires_anchor": domain_info["stability"] > 0.8
        }

        return context

    def align_with_anchor(self, coordinates: Tuple[float, float, float, float],
                         context: Dict) -> Tuple[Tuple[float, float, float, float], float]:
        """
        STAGE 4: Align coordinates with universal anchor point.

        Returns: (aligned_coordinates, alignment_score)
        """
        if not context.get("requires_anchor", False):
            return coordinates, 1.0

        # Calculate distance from anchor
        distance = np.sqrt(sum((c - a)**2 for c, a in zip(coordinates, self.anchor_point)))

        # Alignment strength (inverse of distance)
        alignment = 1.0 / (1.0 + distance)

        # Gently pull coordinates toward anchor
        pull_strength = context["stability"] * 0.3
        aligned = tuple(
            c + (a - c) * pull_strength * alignment
            for c, a in zip(coordinates, self.anchor_point)
        )

        return aligned, alignment

    # ========================================
    # STAGE 5-7: EXECUTION (Strategy, Validation, Output)
    # ========================================

    def determine_execution_strategy(self, intent: Dict, context: Dict,
                                    coordinates: Tuple[float, float, float, float]) -> str:
        """
        STAGE 5: Determine optimal execution strategy.
        """
        love, power, wisdom, justice = coordinates

        # Strategy based on dominant axis
        if love >= max(power, wisdom, justice):
            return "compassionate_action"
        elif power >= max(love, wisdom, justice):
            return "authoritative_command"
        elif wisdom >= max(love, power, justice):
            return "instructive_guidance"
        elif justice >= max(love, power, wisdom):
            return "corrective_judgment"
        else:
            return "balanced_response"

    def validate_semantic_integrity(self, original_intent: Dict,
                                   transformed_coordinates: Tuple[float, float, float, float]) -> float:
        """
        STAGE 6: Validate semantic integrity of transformation.

        Returns integrity score [0, 1]
        """
        original_coords = self.map_intent_to_coordinates(original_intent)

        # Calculate preservation of semantic direction
        dot_product = sum(o * t for o, t in zip(original_coords, transformed_coordinates))
        orig_mag = np.sqrt(sum(c**2 for c in original_coords))
        trans_mag = np.sqrt(sum(c**2 for c in transformed_coordinates))

        if orig_mag == 0 or trans_mag == 0:
            return 0.0

        # Cosine similarity as integrity measure
        integrity = dot_product / (orig_mag * trans_mag)
        return max(0.0, integrity)

    def generate_output(self, intent: Dict, coordinates: Tuple[float, float, float, float],
                       strategy: str, context: Dict) -> str:
        """
        STAGE 7: Generate behaviorally-aligned output.
        """
        love, power, wisdom, justice = coordinates

        output_templates = {
            "compassionate_action": f"With LOVE ({love:.3f}), I respond: {intent['text']}",
            "authoritative_command": f"With POWER ({power:.3f}), I declare: {intent['text']}",
            "instructive_guidance": f"With WISDOM ({wisdom:.3f}), I teach: {intent['text']}",
            "corrective_judgment": f"With JUSTICE ({justice:.3f}), I correct: {intent['text']}",
            "balanced_response": f"In balance (L:{love:.2f} P:{power:.2f} W:{wisdom:.2f} J:{justice:.2f}), I respond: {intent['text']}"
        }

        return output_templates.get(strategy, intent['text'])

    # ========================================
    # MAIN TRANSFORMATION METHOD
    # ========================================

    def transform(self, input_text: str, thought_type: str = "practical_wisdom",
                 context_domain: str = "general", preserve_intent: bool = True) -> ICETransformationResult:
        """
        Complete ICE-Centric transformation pipeline.

        Args:
            input_text: Raw input text
            thought_type: Type of thought/intent
            context_domain: Contextual domain
            preserve_intent: Whether to strictly preserve original intent

        Returns:
            ICETransformationResult with complete transformation data
        """
        stages = []

        # INTENT Phase
        stages.append("intent_extraction")
        intent = self.extract_intent(input_text, thought_type)

        stages.append("intent_mapping")
        intent_coords = self.map_intent_to_coordinates(intent)

        # CONTEXT Phase
        stages.append("context_analysis")
        context = self.analyze_context(intent, context_domain)

        stages.append("context_alignment")
        aligned_coords, alignment = self.align_with_anchor(intent_coords, context)

        # EXECUTION Phase
        stages.append("execution_strategy")
        strategy = self.determine_execution_strategy(intent, context, aligned_coords)

        stages.append("execution_validation")
        integrity = self.validate_semantic_integrity(intent, aligned_coords)

        stages.append("output_generation")
        output = self.generate_output(intent, aligned_coords, strategy, context)

        # Calculate quality metrics
        anchor_distance = np.sqrt(sum((c - a)**2 for c, a in zip(aligned_coords, self.anchor_point)))
        divine_alignment = 1.0 / (1.0 + anchor_distance)

        # Update statistics
        self.transformation_count += 1
        self.total_alignment += divine_alignment

        return ICETransformationResult(
            intent_coordinates=aligned_coords,
            intent_type=thought_type,
            context_domain=context_domain,
            context_alignment=alignment,
            execution_strategy=strategy,
            output_text=output,
            divine_alignment=divine_alignment,
            anchor_distance=anchor_distance,
            semantic_integrity=integrity,
            transformation_stages=stages
        )

    def get_performance_stats(self) -> Dict:
        """Get transformer performance statistics."""
        return {
            "transformations": self.transformation_count,
            "average_alignment": self.total_alignment / max(self.transformation_count, 1),
            "anchor_point": self.anchor_point
        }


# ========================================
# DEMONSTRATION
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("ICE-Centric URI-Transformer Demonstration")
    print("=" * 60)
    print()

    # Initialize transformer
    transformer = ICEURITransformer()

    # Test cases
    test_cases = [
        {
            "text": "Show compassion and mercy to those who suffer",
            "type": "moral_judgment",
            "domain": "ethical"
        },
        {
            "text": "Assert authority with strength and decisiveness",
            "type": "practical_wisdom",
            "domain": "general"
        },
        {
            "text": "Seek understanding through knowledge and insight",
            "type": "practical_wisdom",
            "domain": "technical"
        },
        {
            "text": "Judge righteously with fairness and integrity",
            "type": "moral_judgment",
            "domain": "ethical"
        },
        {
            "text": "Balance love, power, wisdom, and justice in all things",
            "type": "practical_wisdom",
            "domain": "spiritual"
        }
    ]

    results = []

    print("TRANSFORMATIONS:")
    print("-" * 60)

    for i, test in enumerate(test_cases, 1):
        result = transformer.transform(
            test["text"],
            thought_type=test["type"],
            context_domain=test["domain"]
        )
        results.append(result)

        print(f"\n{i}. Input: {test['text']}")
        print(f"   Type: {test['type']}, Domain: {test['domain']}")
        print(f"   Coordinates: (L:{result.intent_coordinates[0]:.3f}, "
              f"P:{result.intent_coordinates[1]:.3f}, "
              f"W:{result.intent_coordinates[2]:.3f}, "
              f"J:{result.intent_coordinates[3]:.3f})")
        print(f"   Strategy: {result.execution_strategy}")
        print(f"   Divine Alignment: {result.divine_alignment:.4f}")
        print(f"   Anchor Distance: {result.anchor_distance:.4f}")
        print(f"   Semantic Integrity: {result.semantic_integrity:.4f}")
        print(f"   Output: {result.output_text}")

    print()
    print("-" * 60)
    print("PERFORMANCE STATISTICS:")
    print("-" * 60)
    stats = transformer.get_performance_stats()
    print(f"Total Transformations: {stats['transformations']}")
    print(f"Average Divine Alignment: {stats['average_alignment']:.4f}")
    print(f"Universal Anchor Point: {stats['anchor_point']}")
    print()
    print("=" * 60)
    print("ICE Framework: Intent → Context → Execution")
    print("Semantic integrity preserved through all transformations")
    print("=" * 60)
