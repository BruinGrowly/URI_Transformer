"""
Principle-Based Knowledge Graph
"""

from dataclasses import dataclass, field
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
            Principle("Benevolent Love", "The principle of selfless, unconditional care.", PhiCoordinate(1.8, 1.4, 1.0, 1.6)),
            Principle("Righteous Justice", "The principle of fairness, truth, and moral order.", PhiCoordinate(1.4, 1.8, 1.2, 1.6)),
            Principle("Effective Power", "The principle of capability, strength, and authority.", PhiCoordinate(1.0, 1.2, 1.8, 1.4)),
            Principle("Discerning Wisdom", "The principle of knowledge, understanding, and insight.", PhiCoordinate(1.6, 1.6, 1.4, 1.8)),
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
