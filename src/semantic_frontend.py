"""
Semantic Front-End
==================

This module provides the crucial first step in the TruthSense pipeline.
It replaces the deterministic hashing function with a lexical, meaning-driven
approach to map raw text to a 4D PhiCoordinate.
"""

from src.phi_geometric_engine import PhiCoordinate


class SemanticFrontEnd:
    """
    A lexical-based semantic front-end that maps text to a PhiCoordinate.
    """
    def __init__(self):
        self.lexical_db = self._initialize_lexical_db()

    def _initialize_lexical_db(self):
        """
        Initializes a curated lexical database.
        Values are tuples of (Love, Justice, Power, Wisdom).
        """
        return {
            # Core Concepts
            "love":       (1.0, 0.7, 0.6, 0.8),
            "justice":    (0.7, 1.0, 0.8, 0.9),
            "power":      (0.6, 0.8, 1.0, 0.7),
            "wisdom":     (0.8, 0.9, 0.7, 1.0),

            # Virtues (Love-dominant)
            "compassion": (0.9, 0.6, 0.5, 0.7),
            "kindness":   (0.9, 0.7, 0.4, 0.6),
            "humility":   (0.8, 0.8, 0.3, 0.9),

            # Principles (Justice-dominant)
            "truth":      (0.7, 0.9, 0.6, 0.9),
            "fairness":   (0.6, 0.9, 0.5, 0.7),
            "integrity":  (0.7, 0.9, 0.7, 0.8),

            # Vices (Low-Justice)
            "hate":       (0.1, 0.2, 0.8, 0.3),
            "deception":  (0.2, 0.1, 0.6, 0.4),
            "lies":       (0.3, 0.1, 0.5, 0.2),
            "unfair":     (0.4, 0.2, 0.6, 0.3),

            # Neutral/Other
            "leader":     (0.6, 0.7, 0.8, 0.8),
            "society":    (0.5, 0.6, 0.5, 0.5),
        }

    def text_to_coordinate(self, text: str) -> PhiCoordinate:
        """
        Analyzes raw text against the lexical database to generate a
        weighted PhiCoordinate.
        """
        words = text.lower().replace('.', '').replace(',', '').split()
        if not words:
            return PhiCoordinate(0.5, 0.5, 0.5, 0.5)

        total_love, total_justice, total_power, total_wisdom = 0.0, 0.0, 0.0, 0.0
        word_count = 0

        for word in words:
            if word in self.lexical_db:
                coords = self.lexical_db[word]
                total_love += coords[0]
                total_justice += coords[1]
                total_power += coords[2]
                total_wisdom += coords[3]
                word_count += 1

        if word_count == 0:
            return PhiCoordinate(0.5, 0.5, 0.5, 0.5)

        # Average the coordinates
        avg_love = total_love / word_count
        avg_justice = total_justice / word_count
        avg_power = total_power / word_count
        avg_wisdom = total_wisdom / word_count

        return PhiCoordinate(avg_love, avg_justice, avg_power, avg_wisdom)
