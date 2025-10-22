"""
TruthSense Transformer (Deep ICE)
=================================

The core of the URI Transformer project, this refactored transformer
implements a deep and integrated 7-stage ICE pipeline, processing
structured objects and leveraging the evolved foundational frameworks.
"""

import hashlib
import numpy as np
from src.phi_geometric_engine import PhiCoordinate, GoldenSpiral, PhiExponentialBinner
from src.frameworks import QLAEFramework, GODFramework, QLAEContext, ExecutionPlan
from src.output_generator import OutputGenerator
from src.data_structures import Intent, TruthSenseResult

class TruthSenseTransformer:
    """The refactored transformer with a deep, integrated ICE pipeline."""

    def __init__(self, anchor_point=PhiCoordinate(1.0, 1.0, 1.0, 1.0)):
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

    def _generate_deterministic_coord(self, text: str) -> PhiCoordinate:
        """Generates a unique, deterministic 4D coordinate from a string."""
        h = hashlib.sha256(text.encode()).hexdigest()
        vals = [int(h[i:i+16], 16) / (16**16 - 1) for i in range(0, 64, 16)]
        return PhiCoordinate(love=vals[0], justice=vals[1], power=vals[2], wisdom=vals[3])

    def transform(self, input_text: str) -> TruthSenseResult:
        """Runs the full deep and integrated ICE pipeline."""

        # 1. Generate Raw Coordinate
        raw_coord = self._generate_deterministic_coord(input_text)

        # 2. Align Coordinate with Anchor
        anchor_distance = self.phi_engine["spiral"].distance(raw_coord, self.anchor_point)
        alignment_strength = 1 / (1 + anchor_distance)

        l = raw_coord.love + (self.anchor_point.love - raw_coord.love) * alignment_strength
        j = raw_coord.justice + (self.anchor_point.justice - raw_coord.justice) * alignment_strength
        p = raw_coord.power + (self.anchor_point.power - raw_coord.power) * alignment_strength
        w = raw_coord.wisdom + (self.anchor_point.wisdom - raw_coord.wisdom) * alignment_strength
        aligned_coord = PhiCoordinate(l, j, p, w)

        # 3. Process through Deep ICE Framework
        # INTENT (L+W)
        intent = Intent(
            purpose=f"To act with benevolent purpose (Love: {aligned_coord.love:.2f})",
            guiding_principles=[f"Guided by wisdom and understanding (Wisdom: {aligned_coord.wisdom:.2f})"]
        )

        # CONTEXT (J)
        context = self.frameworks["qlae"].get_context(aligned_coord)
        # Justice as moderator
        truth_sense_validation = aligned_coord.justice > 0.5
        context.is_valid = truth_sense_validation

        # EXECUTION (P)
        execution = self.frameworks["god"].generate_plan(aligned_coord.power, aligned_coord)

        # 4. Final Output Generation
        final_output = self.output_generator.generate(intent, context, execution)

        # 5. Calculate Metrics
        original_bin = self.phi_engine["binner"].get_bin(np.linalg.norm(raw_coord.to_numpy()))
        aligned_bin = self.phi_engine["binner"].get_bin(np.linalg.norm(aligned_coord.to_numpy()))
        semantic_integrity = 1.0 if original_bin == aligned_bin else 0.9

        # 6. Result Compilation
        return TruthSenseResult(
            raw_coord=raw_coord,
            aligned_coord=aligned_coord,
            intent=intent,
            context=context,
            execution=execution,
            final_output=final_output,
            anchor_distance=anchor_distance,
            semantic_integrity=semantic_integrity,
            truth_sense_validation=truth_sense_validation
        )
