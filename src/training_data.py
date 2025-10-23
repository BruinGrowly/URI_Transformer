"""
Labeled Training Data for the Semantic Front-End
================================================

This module contains a curated list of sentences, each manually labeled
with its corresponding PhiCoordinate (Love, Justice, Power, Wisdom).

This dataset serves as the ground truth for training the projection head
of our hybrid semantic front-end.
"""

TRAINING_DATA = [
    # --- High Love ---
    ("True love is compassionate and kind.", (0.9, 0.7, 0.5, 0.8)),
    ("She showed great compassion for the suffering.", (0.9, 0.6, 0.4, 0.7)),
    ("His kindness was a gift to everyone he met.", (0.9, 0.7, 0.3, 0.6)),

    # --- High Justice ---
    ("A just society is built on fairness and truth.", (0.7, 0.9, 0.6, 0.8)),
    ("He acted with integrity in all his dealings.", (0.6, 0.9, 0.7, 0.8)),
    ("The judge delivered a fair and righteous verdict.", (0.5, 0.9, 0.8, 0.7)),

    # --- High Power ---
    ("The king had absolute power over his domain.", (0.3, 0.5, 0.9, 0.6)),
    ("The storm was a powerful and unstoppable force.", (0.1, 0.3, 0.9, 0.2)),
    ("She spoke with authority and conviction.", (0.4, 0.6, 0.9, 0.7)),

    # --- High Wisdom ---
    ("The philosopher shared his profound wisdom.", (0.6, 0.7, 0.5, 0.9)),
    ("True wisdom is knowing you know nothing.", (0.7, 0.8, 0.4, 0.9)),
    ("She offered wise counsel to the young leader.", (0.7, 0.7, 0.6, 0.9)),

    # --- Low Justice (Vices) ---
    ("His actions were unjust and deceitful.", (0.2, 0.1, 0.6, 0.3)),
    ("The plan was built on a foundation of lies.", (0.3, 0.1, 0.5, 0.2)),
    ("Hate and division are destructive forces.", (0.1, 0.2, 0.8, 0.3)),

    # --- Balanced / Mixed ---
    ("A good leader rules with power, wisdom, and justice.", (0.6, 0.8, 0.8, 0.8)),
    ("Love without truth is sentimentality.", (0.8, 0.5, 0.4, 0.7)),
    ("Power without justice is tyranny.", (0.3, 0.2, 0.9, 0.4)),
]
