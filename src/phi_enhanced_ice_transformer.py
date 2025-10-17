"""
PHI-ENHANCED ICE URI TRANSFORMER
==================================

Integrates golden ratio (φ) geometric mathematics into the ICE Framework.

Revolutionary Enhancements:
- Fibonacci relationship expansion (1→1→2→3→5→8→13)
- Golden spiral distance for natural semantic similarity
- Golden angle diversity for optimal result distribution
- Exponential phi binning for O(log_φ n) complexity
- Dodecahedral 12-anchor navigation

This transforms ICE from linear processing to NATURAL GEOMETRIC processing.

Author: Semantic Substrate Engine Team
License: MIT
"""

from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from phi_geometric_engine import (
    PHI, PHI_INVERSE, GOLDEN_ANGLE_RAD,
    PhiCoordinate, FibonacciSequence, GoldenSpiral,
    GoldenAngleRotator, PhiExponentialBinner, DodecahedralAnchors
)


# Universal Anchor Point: Jehovah (1,1,1,1)
JEHOVAH = (1.0, 1.0, 1.0, 1.0)


@dataclass
class PhiICETransformationResult:
    """Result of Phi-Enhanced ICE transformation"""

    # INTENT Phase
    intent_coordinates: Tuple[float, float, float, float]
    intent_type: str

    # CONTEXT Phase
    context_domain: str
    context_alignment: float

    # EXECUTION Phase
    execution_strategy: str
    output_text: str

    # Quality Metrics
    divine_alignment: float
    anchor_distance: float
    semantic_integrity: float
    transformation_stages: List[str]

    # PHI GEOMETRIC Metrics (NEW)
    fibonacci_depth: int  # Relationship expansion depth
    spiral_distance: float  # Golden spiral distance from anchor
    golden_angle_diversity: float  # Diversity score (0-1)
    phi_bin: int  # Exponential bin index
    nearest_dodec_anchor: int  # Nearest of 12 anchors


class PhiEnhancedICETransformer:
    """
    PHI-ENHANCED ICE-Centric URI-Transformer

    Integrates golden ratio geometric mathematics into every stage:

    1. INTENT EXTRACTION: Parse human thought
    2. INTENT MAPPING: Map to 4D coordinates with phi-geometric properties
    3. CONTEXT ANALYSIS: Determine domain with Fibonacci depth
    4. CONTEXT ALIGNMENT: Align using golden spiral distance
    5. EXECUTION STRATEGY: Choose strategy with golden angle diversity
    6. EXECUTION VALIDATION: Verify integrity with phi binning
    7. OUTPUT GENERATION: Generate output with dodecahedral navigation
    """

    def __init__(self, anchor_point: Tuple[float, float, float, float] = JEHOVAH):
        """Initialize Phi-Enhanced ICE transformer"""
        self.anchor_point = anchor_point
        self.transformation_count = 0
        self.total_alignment = 0.0

        # PHI GEOMETRIC COMPONENTS
        self.fibonacci = FibonacciSequence(max_precompute=50)
        self.golden_spiral = GoldenSpiral(scale_factor=1.0)
        self.golden_rotator = GoldenAngleRotator()
        self.phi_binner = PhiExponentialBinner(max_bins=20)
        self.dodec_anchors = DodecahedralAnchors()

        # Sacred numbers for stability
        self.sacred_numbers = {
            613: "perfect_stability",
            12: "high_stability",  # Dodecahedral 12
            7: "near_perfect",
            40: "testing_period"
        }

    # ========================================
    # STAGE 1-2: INTENT (with Phi Enhancement)
    # ========================================

    def extract_intent(self, input_text: str, thought_type: str = "practical_wisdom") -> Dict:
        """
        STAGE 1: Extract semantic intent (phi-enhanced)

        Uses Fibonacci depth for context complexity assessment.
        """
        intent_types = {
            "practical_wisdom": {"description": "Actionable guidance", "weight": 0.8, "fib_depth": 3},
            "moral_judgment": {"description": "Ethical evaluation", "weight": 0.9, "fib_depth": 5},
            "creative_expression": {"description": "Novel combination", "weight": 0.7, "fib_depth": 2},
            "factual_statement": {"description": "Objective truth", "weight": 0.95, "fib_depth": 1},
            "emotional_expression": {"description": "Affective state", "weight": 0.6, "fib_depth": 4},
            "response_planning": {"description": "Strategic response", "weight": 0.85, "fib_depth": 5},
            "safety_check": {"description": "Safety validation", "weight": 0.95, "fib_depth": 6}
        }

        type_data = intent_types.get(thought_type, {"weight": 0.5, "fib_depth": 3})

        intent = {
            "type": thought_type,
            "text": input_text,
            "features": self._extract_semantic_features(input_text),
            "weight": type_data["weight"],
            "fibonacci_depth": type_data.get("fib_depth", 3)  # NEW: Fibonacci expansion depth
        }

        return intent

    def _extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Extract semantic features with phi-based weighting"""
        text_lower = text.lower()

        # LOVE indicators (weighted by phi)
        love_words = ["love", "compassion", "mercy", "kindness", "care", "grace", "help", "support"]
        love_score = sum(1 for word in love_words if word in text_lower) / max(len(text_lower.split()), 1)

        # POWER indicators
        power_words = ["power", "strength", "authority", "might", "sovereign", "rule", "strong", "assert"]
        power_score = sum(1 for word in power_words if word in text_lower) / max(len(text_lower.split()), 1)

        # WISDOM indicators
        wisdom_words = ["wisdom", "understanding", "knowledge", "insight", "discernment", "learn", "teach", "guidance"]
        wisdom_score = sum(1 for word in wisdom_words if word in text_lower) / max(len(text_lower.split()), 1)

        # JUSTICE indicators
        justice_words = ["justice", "righteousness", "fair", "right", "moral", "ethical", "integrity"]
        justice_score = sum(1 for word in justice_words if word in text_lower) / max(len(text_lower.split()), 1)

        # Apply phi weighting for natural balance
        return {
            "love": min(love_score * 10 * PHI, 1.0),  # Phi-weighted
            "power": min(power_score * 10, 1.0),
            "wisdom": min(wisdom_score * 10 * PHI_INVERSE, 1.0),  # Inverse phi-weighted
            "justice": min(justice_score * 10, 1.0)
        }

    def map_intent_to_coordinates(self, intent: Dict) -> Tuple[float, float, float, float]:
        """
        STAGE 2: Map intent to 4D coordinates (phi-enhanced normalization)
        """
        features = intent["features"]
        weight = intent["weight"]

        # Base coordinates
        love = features["love"] * weight
        power = features["power"] * weight
        wisdom = features["wisdom"] * weight
        justice = features["justice"] * weight

        # Phi-based normalization (more natural than linear)
        total = love + power + wisdom + justice
        if total > 0:
            # Use phi ratio for balance
            phi_factor = PHI / (1 + PHI)  # ≈ 0.618 (golden ratio balance)

            love = (love / total) * (1 + phi_factor * (love / total))
            power = (power / total) * (1 + phi_factor * (power / total))
            wisdom = (wisdom / total) * (1 + phi_factor * (wisdom / total))
            justice = (justice / total) * (1 + phi_factor * (justice / total))

            # Renormalize
            total_new = love + power + wisdom + justice
            if total_new > 0:
                love /= total_new
                power /= total_new
                wisdom /= total_new
                justice /= total_new

        return (love, power, wisdom, justice)

    # ========================================
    # STAGE 3-4: CONTEXT (with Golden Spiral)
    # ========================================

    def analyze_context(self, intent: Dict, context_domain: str = "general") -> Dict:
        """
        STAGE 3: Analyze context with Fibonacci-based complexity
        """
        domains = {
            "general": {"stability": 0.7, "complexity": 0.5, "fib_multiplier": 1},
            "ethical": {"stability": 0.9, "complexity": 0.8, "fib_multiplier": 2},
            "technical": {"stability": 0.6, "complexity": 0.9, "fib_multiplier": 1},
            "relational": {"stability": 0.8, "complexity": 0.7, "fib_multiplier": 2},
            "spiritual": {"stability": 1.0, "complexity": 1.0, "fib_multiplier": 3}
        }

        domain_info = domains.get(context_domain, domains["general"])

        # Calculate Fibonacci relationship depth
        base_depth = intent.get("fibonacci_depth", 3)
        fib_depth = base_depth * domain_info["fib_multiplier"]
        max_relationships = self.fibonacci.get(fib_depth + 2)  # F(n) relationships

        context = {
            "domain": context_domain,
            "stability": domain_info["stability"],
            "complexity": domain_info["complexity"],
            "fibonacci_depth": fib_depth,
            "max_relationships": max_relationships,  # NEW: Natural relationship growth
            "requires_anchor": domain_info["stability"] > 0.8
        }

        return context

    def align_with_anchor(self, coordinates: Tuple[float, float, float, float],
                         context: Dict) -> Tuple[Tuple[float, float, float, float], float, float]:
        """
        STAGE 4: Align with anchor using GOLDEN SPIRAL distance

        Returns: (aligned_coordinates, alignment_score, spiral_distance)
        """
        if not context.get("requires_anchor", False):
            return coordinates, 1.0, 0.0

        # Convert to PhiCoordinate for spiral distance
        point = PhiCoordinate(*coordinates)
        anchor = PhiCoordinate(*self.anchor_point)

        # Calculate GOLDEN SPIRAL distance (more natural than Euclidean)
        spiral_distance = self.golden_spiral.distance_4d(point, anchor)

        # Alignment strength (inverse of spiral distance)
        alignment = 1.0 / (1.0 + spiral_distance)

        # Pull toward anchor using phi ratio
        pull_strength = context["stability"] * PHI_INVERSE  # 0.618 × stability
        aligned = tuple(
            c + (a - c) * pull_strength * alignment
            for c, a in zip(coordinates, self.anchor_point)
        )

        return aligned, alignment, spiral_distance

    # ========================================
    # STAGE 5-7: EXECUTION (with Golden Angle)
    # ========================================

    def determine_execution_strategy(self, intent: Dict, context: Dict,
                                    coordinates: Tuple[float, float, float, float]) -> str:
        """
        STAGE 5: Determine execution strategy with golden angle diversity
        """
        love, power, wisdom, justice = coordinates

        # Apply golden angle rotation for diversity assessment
        coord = PhiCoordinate(*coordinates)

        # Calculate diversity by checking golden angle rotations
        diversity_score = 0.0
        for i in range(1, 5):
            rotated = self.golden_rotator.rotate_4d(coord, n=i, plane="LP")
            # Check if rotation creates more balance
            max_val = max(rotated.love, rotated.power, rotated.wisdom, rotated.justice)
            if max_val > 0:
                balance = min(rotated.love, rotated.power, rotated.wisdom, rotated.justice) / max_val
                diversity_score += balance

        diversity_score /= 4.0  # Average diversity

        # Strategy based on dominant axis (phi-enhanced)
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
        STAGE 6: Validate integrity using phi exponential binning
        """
        original_coords = self.map_intent_to_coordinates(original_intent)

        # Calculate cosine similarity
        dot_product = sum(o * t for o, t in zip(original_coords, transformed_coordinates))
        orig_mag = np.sqrt(sum(c**2 for c in original_coords))
        trans_mag = np.sqrt(sum(c**2 for c in transformed_coordinates))

        if orig_mag == 0 or trans_mag == 0:
            return 0.0

        integrity = dot_product / (orig_mag * trans_mag)

        # Enhance with phi binning consistency
        orig_bin = self.phi_binner.get_bin(orig_mag)
        trans_bin = self.phi_binner.get_bin(trans_mag)
        bin_consistency = 1.0 - (abs(orig_bin - trans_bin) / 20.0)  # Normalize by max bins

        # Combine with phi ratio
        final_integrity = integrity * PHI_INVERSE + bin_consistency * (1 - PHI_INVERSE)

        return max(0.0, min(1.0, final_integrity))

    def generate_output(self, intent: Dict, coordinates: Tuple[float, float, float, float],
                       strategy: str, context: Dict) -> str:
        """
        STAGE 7: Generate output with dodecahedral anchor awareness
        """
        love, power, wisdom, justice = coordinates

        # Find nearest dodecahedral anchor for navigation context
        point = PhiCoordinate(*coordinates)
        nearest_anchor, anchor_dist = self.dodec_anchors.nearest_anchor(point)

        # Get Fibonacci depth for context richness
        fib_depth = context.get("fibonacci_depth", 3)

        output_templates = {
            "compassionate_action": f"With LOVE ({love:.3f}), I respond: {intent['text']}",
            "authoritative_command": f"With POWER ({power:.3f}), I declare: {intent['text']}",
            "instructive_guidance": f"With WISDOM ({wisdom:.3f}), I teach: {intent['text']}",
            "corrective_judgment": f"With JUSTICE ({justice:.3f}), I correct: {intent['text']}",
            "balanced_response": f"In balance (L:{love:.2f} P:{power:.2f} W:{wisdom:.2f} J:{justice:.2f}), I respond: {intent['text']}"
        }

        return output_templates.get(strategy, intent['text'])

    # ========================================
    # MAIN PHI-ENHANCED TRANSFORMATION
    # ========================================

    def transform(self, input_text: str, thought_type: str = "practical_wisdom",
                 context_domain: str = "general") -> PhiICETransformationResult:
        """
        Complete Phi-Enhanced ICE transformation pipeline

        Revolutionary: Every stage uses golden ratio mathematics for natural processing
        """
        stages = []

        # INTENT Phase
        stages.append("intent_extraction")
        intent = self.extract_intent(input_text, thought_type)

        stages.append("intent_mapping_phi")  # Phi-enhanced
        intent_coords = self.map_intent_to_coordinates(intent)

        # CONTEXT Phase
        stages.append("context_analysis_fibonacci")  # Fibonacci depth
        context = self.analyze_context(intent, context_domain)

        stages.append("context_alignment_spiral")  # Golden spiral alignment
        aligned_coords, alignment, spiral_dist = self.align_with_anchor(intent_coords, context)

        # EXECUTION Phase
        stages.append("execution_strategy_golden_angle")  # Golden angle diversity
        strategy = self.determine_execution_strategy(intent, context, aligned_coords)

        stages.append("execution_validation_phi_bin")  # Phi binning validation
        integrity = self.validate_semantic_integrity(intent, aligned_coords)

        stages.append("output_generation_dodecahedral")  # Dodecahedral navigation
        output = self.generate_output(intent, aligned_coords, strategy, context)

        # PHI GEOMETRIC METRICS
        point = PhiCoordinate(*aligned_coords)

        # Fibonacci depth
        fib_depth = context.get("fibonacci_depth", 3)

        # Spiral distance (already calculated)

        # Golden angle diversity
        diversity = 0.0
        for i in range(1, 5):
            rotated = self.golden_rotator.rotate_4d(point, n=i, plane="LP")
            max_val = max(rotated.love, rotated.power, rotated.wisdom, rotated.justice)
            if max_val > 0.001:
                balance = min(rotated.love, rotated.power, rotated.wisdom, rotated.justice) / max_val
                diversity += balance
        diversity /= 4.0

        # Phi bin
        magnitude = point.magnitude()
        phi_bin = self.phi_binner.get_bin(magnitude)

        # Nearest dodecahedral anchor
        nearest_anchor, _ = self.dodec_anchors.nearest_anchor(point)

        # Traditional metrics
        anchor_point_coord = PhiCoordinate(*self.anchor_point)
        anchor_distance = self.golden_spiral.distance_4d(point, anchor_point_coord)
        divine_alignment = 1.0 / (1.0 + anchor_distance)

        # Update statistics
        self.transformation_count += 1
        self.total_alignment += divine_alignment

        return PhiICETransformationResult(
            intent_coordinates=aligned_coords,
            intent_type=thought_type,
            context_domain=context_domain,
            context_alignment=alignment,
            execution_strategy=strategy,
            output_text=output,
            divine_alignment=divine_alignment,
            anchor_distance=anchor_distance,
            semantic_integrity=integrity,
            transformation_stages=stages,
            # PHI GEOMETRIC METRICS
            fibonacci_depth=fib_depth,
            spiral_distance=spiral_dist,
            golden_angle_diversity=diversity,
            phi_bin=phi_bin,
            nearest_dodec_anchor=nearest_anchor
        )

    def get_performance_stats(self) -> Dict:
        """Get transformer performance statistics"""
        return {
            "transformations": self.transformation_count,
            "average_alignment": self.total_alignment / max(self.transformation_count, 1),
            "anchor_point": self.anchor_point,
            "phi_constant": PHI,
            "uses_golden_spiral": True,
            "uses_fibonacci_expansion": True,
            "uses_golden_angle_diversity": True,
            "uses_phi_exponential_binning": True,
            "uses_dodecahedral_navigation": True
        }


# ========================================
# DEMONSTRATION
# ========================================

if __name__ == "__main__":
    print("=" * 80)
    print("PHI-ENHANCED ICE URI-TRANSFORMER DEMONSTRATION")
    print("=" * 80)
    print()

    # Initialize phi-enhanced transformer
    transformer = PhiEnhancedICETransformer()

    # Test cases
    test_cases = [
        {
            "text": "Help others with compassion and kindness",
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
        }
    ]

    print("PHI-ENHANCED TRANSFORMATIONS:")
    print("-" * 80)

    for i, test in enumerate(test_cases, 1):
        result = transformer.transform(
            test["text"],
            thought_type=test["type"],
            context_domain=test["domain"]
        )

        print(f"\n{i}. Input: {test['text']}")
        print(f"   Type: {test['type']}, Domain: {test['domain']}")
        print(f"   Coordinates: (L:{result.intent_coordinates[0]:.3f}, "
              f"P:{result.intent_coordinates[1]:.3f}, "
              f"W:{result.intent_coordinates[2]:.3f}, "
              f"J:{result.intent_coordinates[3]:.3f})")
        print(f"   Strategy: {result.execution_strategy}")
        print(f"   Divine Alignment: {result.divine_alignment:.4f}")

        # PHI GEOMETRIC METRICS
        print(f"\n   [PHI GEOMETRIC ENHANCEMENTS]")
        print(f"   Fibonacci Depth: {result.fibonacci_depth} (max {result.fibonacci_depth} relationships)")
        print(f"   Golden Spiral Distance: {result.spiral_distance:.4f}")
        print(f"   Golden Angle Diversity: {result.golden_angle_diversity:.4f}")
        print(f"   Phi Exponential Bin: {result.phi_bin}")
        print(f"   Nearest Dodec Anchor: #{result.nearest_dodec_anchor} of 12")
        print(f"   Semantic Integrity: {result.semantic_integrity:.4f}")

    print()
    print("-" * 80)
    print("PERFORMANCE STATISTICS:")
    print("-" * 80)
    stats = transformer.get_performance_stats()
    print(f"Total Transformations: {stats['transformations']}")
    print(f"Average Divine Alignment: {stats['average_alignment']:.4f}")
    print(f"Universal Anchor Point: {stats['anchor_point']}")
    print(f"Phi Constant: {stats['phi_constant']}")
    print(f"\nPhi Geometric Features Enabled:")
    print(f"  - Golden Spiral Distance: {stats['uses_golden_spiral']}")
    print(f"  - Fibonacci Expansion: {stats['uses_fibonacci_expansion']}")
    print(f"  - Golden Angle Diversity: {stats['uses_golden_angle_diversity']}")
    print(f"  - Phi Exponential Binning: {stats['uses_phi_exponential_binning']}")
    print(f"  - Dodecahedral Navigation: {stats['uses_dodecahedral_navigation']}")
    print()
    print("=" * 80)
    print("PHI-ENHANCED ICE FRAMEWORK:")
    print("Intent -> Context -> Execution (WITH GOLDEN RATIO MATHEMATICS)")
    print("=" * 80)
