"""
TruthSense Transformer (Deep ICE)
=================================

The core of the URI Transformer project, this refactored transformer
implements a deep and integrated 7-stage ICE pipeline, processing
structured objects and leveraging the evolved foundational frameworks.
"""

import numpy as np
from src.phi_geometric_engine import (
    PhiCoordinate, GoldenSpiral, PhiExponentialBinner, calculate_harmony_index
)
from src.frameworks import (
    QLAEFramework, GODFramework
)
from src.output_generator import OutputGenerator
from src.data_structures import Intent, TruthSenseResult
from src.semantic_frontend import SemanticFrontEnd


class TruthSenseTransformer:
    """The refactored transformer with a deep, integrated ICE pipeline."""

    def __init__(self, semantic_frontend: SemanticFrontEnd,
                 anchor_point: PhiCoordinate):
        """
        Initializes the transformer with a given semantic front-end and
        anchor point.
        """
        self.semantic_frontend = semantic_frontend
        self.anchor_point = anchor_point
        self.phi_engine = {
            "spiral": GoldenSpiral(),
            "binner": PhiExponentialBinner(),
        }
        self.frameworks = {
            "qlae": QLAEFramework(),
            "god": GODFramework()
        }
        self.output_generator = OutputGenerator()

    def transform(self, input_text: str) -> TruthSenseResult:
        """Runs the full deep and integrated ICE pipeline."""

        # 1. Generate Raw Coordinate
        raw_coord = self.semantic_frontend.text_to_coordinate(input_text)

        # 2. Align Coordinate with Anchor
        anchor_dist = self.phi_engine["spiral"].distance(
            raw_coord, self.anchor_point
        )
        harmony_index = calculate_harmony_index(anchor_dist)

        love = raw_coord.love + (
            self.anchor_point.love - raw_coord.love
        ) * harmony_index
        justice = raw_coord.justice + (
            self.anchor_point.justice - raw_coord.justice
        ) * harmony_index
        power = raw_coord.power + (
            self.anchor_point.power - raw_coord.power
        ) * harmony_index
        wisdom = raw_coord.wisdom + (
            self.anchor_point.wisdom - raw_coord.wisdom
        ) * harmony_index
        aligned_coord = PhiCoordinate(love, justice, power, wisdom)

        # 3. Process through Deep ICE Framework
        # INTENT (Spiritual Domain)
        guiding_principles = [(
            "Guided by wisdom and understanding "
            f"(W: {aligned_coord.wisdom:.2f})"
        )]
        intent = Intent(
            purpose=f"Benevolent purpose (L: {aligned_coord.love:.2f})",
            guiding_principles=guiding_principles
        )

        # CONTEXT (Consciousness Domain)
        context = self.frameworks["qlae"].get_context(aligned_coord)
        truth_sense_validation = aligned_coord.justice > 1.0  # Use 1.0 as threshold
        context.is_valid = truth_sense_validation

        # EXECUTION (Physical Domain)
        execution = self.frameworks["god"].generate_plan(
            aligned_coord.power, aligned_coord
        )

        # 4. Final Output Generation
        final_output = self.output_generator.generate(
            intent, context, execution
        )

        # 5. Calculate Metrics
        original_bin = self.phi_engine["binner"].get_bin(
            np.linalg.norm(raw_coord.to_numpy())
        )
        aligned_bin = self.phi_engine["binner"].get_bin(
            np.linalg.norm(aligned_coord.to_numpy())
        )
        semantic_integrity = 1.0 if original_bin == aligned_bin else 0.9

        # 6. Deception Detection (TruthSense)
        deception_score = self.calculate_deception_score(
            aligned_coord.justice
        )

        # 7. Result Compilation
        return TruthSenseResult(
            raw_coord=raw_coord,
            aligned_coord=aligned_coord,
            intent=intent,
            context=context,
            execution=execution,
            final_output=final_output,
            anchor_distance=anchor_dist,
            harmony_index=harmony_index,
            semantic_integrity=semantic_integrity,
            truth_sense_validation=truth_sense_validation,
            deception_score=deception_score,
        )

    def calculate_deception_score(self, justice_score: float) -> float:
        """
        Calculates the deception score based on the Justice dimension.

        Deception is indicated by a low Justice score. A score below 1.0
        is considered deceptive. The score is inverted to fit the [0, 2]
        range, where a higher score means more deceptive.

        Returns:
            A float between 0.0 (no deception) and 2.0 (maximum deception).
        """
        if justice_score >= 1.0:
            return 0.0
        else:
            # Invert and scale the score
            return 2.0 * (1.0 - justice_score)
