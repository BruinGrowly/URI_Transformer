"""
Principle-Based Knowledge Graph
"""

from dataclasses import dataclass
from typing import List
from src.phi_geometric_engine import PhiCoordinate, GoldenSpiral


@dataclass
class Principle:
    """A single, axiomatic principle in the knowledge graph."""
    name: str
    description: str
    coordinate: PhiCoordinate


class KnowledgeGraph:
    """Manages and queries a set of foundational principles."""
    def __init__(self):
        self.principles: List[Principle] = []
        self._initialize_graph()

    def _initialize_graph(self):
        """Populates the graph with a set of foundational principles."""
        self.principles.extend([
            # Core Principles
            Principle("Benevolent Love",
                      "The principle of selfless, unconditional care.",
                      PhiCoordinate(1.8, 1.4, 1.0, 1.6)),
            Principle("Righteous Justice",
                      "The principle of fairness, truth, and moral order.",
                      PhiCoordinate(1.4, 1.8, 1.2, 1.6)),
            Principle("Effective Power",
                      "The principle of capability, strength, and authority.",
                      PhiCoordinate(1.0, 1.2, 1.8, 1.4)),
            Principle(
                "Discerning Wisdom",
                "The principle of knowledge, understanding, and insight.",
                PhiCoordinate(1.6, 1.6, 1.4, 1.8)
            ),

            # New, nuanced principles
            Principle(
                "Compassionate Justice",
                "The principle of applying fairness and moral order with "
                "empathy and a desire to heal.",
                PhiCoordinate(1.7, 1.7, 1.3, 1.5)
            ),
            Principle(
                "Strategic Power",
                "The principle of applying strength and authority with "
                "foresight, planning, and understanding.",
                PhiCoordinate(1.3, 1.5, 1.7, 1.7)
            ),
            Principle(
                "Creative Expression",
                "The principle of bringing new ideas and beauty into "
                "existence through skill and imagination.",
                PhiCoordinate(1.7, 1.4, 1.4, 1.7)
            ),
            Principle(
                "Unwavering Integrity",
                "The principle of steadfast adherence to moral and ethical "
                "truths, even in the face of opposition.",
                PhiCoordinate(1.4, 1.8, 1.5, 1.7)
            ),
            Principle(
                "Servant Leadership",
                "The principle of leading by empowering and serving others, "
                "rather than commanding them.",
                PhiCoordinate(1.8, 1.5, 1.4, 1.7)
            ),
        ])

    def find_closest_principle(self, coord: PhiCoordinate) -> Principle:
        """Finds the principle in the graph closest to a given coordinate."""
        min_dist = float('inf')
        closest_principle = None
        spiral = GoldenSpiral()

        for principle in self.principles:
            dist = spiral.distance(coord, principle.coordinate)
            if dist < min_dist:
                min_dist = dist
                closest_principle = principle

        return closest_principle
