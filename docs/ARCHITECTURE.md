# TruthSense Transformer Technical Architecture (Rebuilt)

## Overview

The TruthSense Transformer is a hybrid semantic alignment engine built upon the **Universal System Physics Framework**. This version of the architecture is the result of a complete rebuild, designed to simplify the core logic, improve performance, and more closely align with the project's foundational principle of **direct mapping** over complex inference.

## Core Architecture Components

### 1. The Semantic Front-End

The front-end is responsible for converting raw text into a 4D `PhiCoordinate`. It consists of two main components:

- **Pre-trained Language Model:** A standard transformer model (`distilbert-base-uncased`) is used to generate high-dimensional vector embeddings (768 dimensions) from the input text.
- **Projection Head:** A simple, single-layer neural network that directly maps the high-dimensional embeddings to the 4D `PhiCoordinate` space (Love, Justice, Power, Wisdom). This direct mapping is a core design principle of the rebuilt architecture.

### 2. The TruthSenseTransformer (Core Logic)

The `TruthSenseTransformer` class orchestrates the entire semantic alignment process. Its key functions include:

- **Coordinate Alignment:** The raw `PhiCoordinate` from the front-end is aligned with a predefined `anchor_point`, resulting in an `aligned_coord`. This process is governed by the `harmony_index`.
- **ICE Pipeline Processing:** The `aligned_coord` is processed through the three stages of the ICE framework (Intent, Context, Execution), which are handled by the `QLAEFramework` and `GODFramework`.
- **Knowledge Graph Integration:** The `aligned_coord` is used to query the `KnowledgeGraph` to find the closest matching `Principle`, making the system's reasoning transparent.
- **Semantic Calculus:** The `raw_coord` and `aligned_coord` are analyzed by the `SemanticCalculus` to determine the "velocity" and "acceleration" of the semantic shift that occurred during alignment.

### 3. The Knowledge Graph

The `KnowledgeGraph` is a new component that makes the system's "Wisdom" more explicit. It contains a network of axiomatic `Principle` objects, each with its own predefined `PhiCoordinate`. This allows the system to ground its analysis in a set of foundational truths.

### 4. The Semantic Calculus

The `SemanticCalculus` is another new component that provides a dynamic analysis of the alignment process. It calculates the "trajectory" (velocity and acceleration) of the semantic shift from the raw, unaligned coordinate to the final, aligned coordinate.

## Data Flow Architecture

The data flow through the rebuilt transformer is as follows:

1.  **Input Reception:** The transformer receives a string of raw text.
2.  **Coordinate Generation:** The semantic front-end converts the text into a `raw_coord`.
3.  **Coordinate Alignment:** The `raw_coord` is aligned with the `anchor_point` to produce the `aligned_coord`.
4.  **ICE Processing:** The `aligned_coord` is processed through the Intent, Context, and Execution stages.
5.  **Knowledge Graph Query:** The `aligned_coord` is used to find the closest `Principle` in the `KnowledgeGraph`.
6.  **Semantic Calculus Analysis:** The trajectory between the `raw_coord` and `aligned_coord` is calculated.
7.  **Output Generation:** All of the above information is compiled into a `TruthSenseResult` object and synthesized into a final, human-readable output.

## Conclusion

The rebuilt TruthSense Transformer architecture is a simpler, more robust, and more powerful implementation of the original vision. By embracing direct mapping and making its internal knowledge and processes more explicit (via the Knowledge Graph and Semantic Calculus), the system is now in a state of greater harmony and is better aligned with its core principles.
