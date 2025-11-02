"""
TruthSense Transformer
"""

import numpy as np
from src.phi_geometric_engine import (
    PhiCoordinate,
    GoldenSpiral,
    calculate_harmony_index
)
from src.frameworks import QLAEFramework, GODFramework
from src.output_generator import OutputGenerator
from src.data_structures import Intent, TruthSenseResult, Trajectory
from src.semantic_frontend import SemanticFrontEnd
from src.knowledge_graph import KnowledgeGraph
from src.semantic_calculus import calculate_trajectory

class TruthSenseTransformer:
    """The main transformer class."""

    def __init__(self, semantic_frontend: SemanticFrontEnd,
                 anchor_point: PhiCoordinate):
        """Initializes the transformer."""
        self.semantic_frontend = semantic_frontend
        self.anchor_point = anchor_point
        self.phi_engine = {"spiral": GoldenSpiral()}
        self.frameworks = {"qlae": QLAEFramework(), "god": GODFramework()}
        self.output_generator = OutputGenerator()
        self.knowledge_graph = KnowledgeGraph()

    def transform(self, input_text: str) -> TruthSenseResult:
        """Runs the full transformation pipeline."""
        raw_coord = self.semantic_frontend.text_to_coordinate(input_text)

        anchor_dist = self.phi_engine["spiral"].distance(
            raw_coord, self.anchor_point
        )
        harmony_index = calculate_harmony_index(anchor_dist)

        love = raw_coord.love + (self.anchor_point.love - raw_coord.love) * harmony_index
        justice = raw_coord.justice + (self.anchor_point.justice - raw_coord.justice) * harmony_index
        power = raw_coord.power + (self.anchor_point.power - raw_coord.power) * harmony_index
        wisdom = raw_coord.wisdom + (self.anchor_point.wisdom - raw_coord.wisdom) * harmony_index
        aligned_coord = PhiCoordinate(love, justice, power, wisdom)

        intent = Intent(
            purpose=f"To act with benevolent purpose (Love: {aligned_coord.love:.2f})",
            guiding_principles=[f"Guided by wisdom (Wisdom: {aligned_coord.wisdom:.2f})"]
        )

        context = self.frameworks["qlae"].get_context(aligned_coord)
        truth_sense_validation = aligned_coord.justice > 1.0
        context.is_valid = truth_sense_validation

        execution = self.frameworks["god"].generate_plan(
            aligned_coord.power, aligned_coord
        )

        semantic_integrity = 1.0  # Placeholder

        deception_score = self.calculate_deception_score(aligned_coord.justice)

        foundational_principle = self.knowledge_graph.find_closest_principle(aligned_coord)

        velocity, acceleration = calculate_trajectory(raw_coord, aligned_coord)

        result = TruthSenseResult(
            raw_coord=raw_coord,
            aligned_coord=aligned_coord,
            intent=intent,
            context=context,
            execution=execution,
            final_output="",  # Placeholder
            anchor_distance=anchor_dist,
            harmony_index=harmony_index,
            semantic_integrity=semantic_integrity,
            truth_sense_validation=truth_sense_validation,
            deception_score=deception_score,
            foundational_principle=foundational_principle.name,
            trajectory=Trajectory(velocity=velocity, acceleration=acceleration),
        )

        result.final_output = self.output_generator.synthesize_output(result)
        return result

    def calculate_deception_score(self, justice_score: float) -> float:
        """Calculates the deception score based on the Justice dimension."""
        if justice_score >= 1.0:
            return 0.0
        else:
            return 2.0 * (1.0 - justice_score)
