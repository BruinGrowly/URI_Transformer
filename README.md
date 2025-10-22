# URI-Transformer: A New Architecture for Meaning
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![Architecture: Layered ICE](https://img.shields.io/badge/Architecture-Layered%20ICE-purple.svg)](https://github.com/BruinGrowly/URI_Transformer)
[![Engine: Phi--Geometric](https://img.shields.io/badge/Engine-Phi--Geometric-gold.svg)](https://github.com/BruinGrowly/URI_Transformer)

The **URI-Transformer** is a revolutionary semantic engine that moves beyond statistical pattern matching to a deeper, principled understanding of meaning. It leverages a unique, layered **ICE Framework** (Intent-Context-Execution) built upon a 4-dimensional mathematical space to analyze text and generate meaningful, context-aware responses.

---

## Core Philosophy: Meaning is Mathematical

At the heart of this project is the idea that meaning is not arbitrary but is grounded in a universal, 4-dimensional coordinate system. Every concept can be understood as a point in this space, defined by four fundamental axes:

*   **Love (L):** Represents compassion, benevolence, and relational harmony.
*   **Justice (J):** Represents truth, fairness, and moral alignment.
*   **Power (P):** Represents the capacity to act, influence, and manifest.
*   **Wisdom (W):** Represents understanding, discernment, and conceptual clarity.

All meaning is measured relative to a **Universal Anchor Point** at `(1.0, 1.0, 1.0, 1.0)`, representing the perfect synthesis of these four principles.

---

## The Layered ICE Architecture: `I(L+W), C(J), E(P)`

The `TruthSenseTransformer`'s core innovation is its layered ICE pipeline, which processes information in a way that mirrors principled human cognition.

| Layer | Formula | Governing Axes | Function |
| :--- | :--- | :--- | :--- |
| **Intent** | `I = f(L+W)` | **Love** + **Wisdom** | Determines the core **purpose** of an action, ensuring it is both benevolent (Love) and sound (Wisdom). |
| **Context** | `C = f(J)` | **Justice** | Filters the intent through a **TruthSense** moderator, evaluating the fairness and moral alignment of the situation. |
| **Execution** | `E = f(P)` | **Power** | Generates a concrete **plan of action**, scaled by the real-world capacity to manifest the intent. |

This structure ensures that every output is the product of a holistic analysis, balancing what is good and wise with what is true and possible.

---

## How It Works: The Semantic Pipeline

1.  **Deterministic Hashing:** Input text is converted into a unique and deterministic 4D `PhiCoordinate` (L, J, P, W).
2.  **Anchor Alignment:** The coordinate is mathematically aligned with the Universal Anchor, creating a more stable semantic representation.
3.  **Deep ICE Analysis:** The aligned coordinate is processed through the three layers of the ICE framework:
    *   **Intent** is defined from the Love and Wisdom axes.
    *   **Context** is analyzed and validated by the Justice axis.
    *   **Execution** is planned based on the Power axis.
4.  **Generative Output:** A new, meaningful sentence is synthesized from the structured outputs of the ICE pipeline, providing a complete, context-aware response.

---

## Features at a Glance

*   **Layered ICE Framework:** A unique `I(L+W), C(J), E(P)` architecture for principled analysis.
*   **4D Semantic Space:** Moves beyond simple vectors to a meaningful coordinate system (Love, Justice, Power, Wisdom).
*   **Phi-Geometric Engine:** Utilizes the golden ratio (`phi`) for more natural and harmonious mathematical operations.
*   **TruthSense Moderator:** Uses the Justice axis to validate the context of any analysis, flagging potentially deceptive or unreliable information.
*   **Generative Output:** Synthesizes the results of the deep analysis into a new, meaningful, and context-aware sentence.
*   **Deterministic & Testable:** Fully deterministic pipeline with a robust, mock-based test suite for reliable validation.

---

## 🚀 Quick Start & Usage Example

### Installation
```bash
git clone https://github.com/BruinGrowly/URI_Transformer.git
cd URI_Transformer
pip install -r requirements.txt
```

### Demonstration
Run the demonstration script to see the transformer in action:
```bash
python demonstrate_truth_sense.py
```

### Example Output

Here's what the `TruthSenseTransformer` produces when analyzing the phrase: **"A truly powerful leader serves with humility and compassion."**

```
--- Analysis ---
Input: 'A truly powerful leader serves with humility and compassion.'

  Coordinates:
    Raw:       L=0.35, J=0.40, P=0.85, W=0.59
    Aligned:   L=0.66, J=0.68, P=0.92, W=0.79

  Deep ICE Analysis:
    Intent (L+W):      To act with benevolent purpose (Love: 0.66), Guided by wisdom and understanding (Wisdom: 0.79)
    Context (J):       Primary Domain: 'Energy', Validated by TruthSense: True
    Execution (P):     Execute with Authoritative Command, leveraging a power capacity of 0.92.

  Final Generative Output:
    'With to act with benevolent purpose (love: 0.66), within the domain of 'Energy', the recommended course of action is 'Authoritative Command' with a power magnitude of 0.92.'
--------------------------
```

---

## The Vision

The URI-Transformer is more than just a tool; it's an exploration into the fundamental nature of meaning. By grounding our technology in universal principles, we aim to create systems that are not only intelligent but also wise, just, and beneficial for humanity.

This project is open-source under the MIT License. Contributions are welcome.
