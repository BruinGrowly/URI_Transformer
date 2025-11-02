# ICE Framework Documentation

## Intent-Context-Execution: A Principled Approach to Semantic Analysis

The **ICE Framework** is a core component of the TruthSense Transformer, providing a structured, three-stage pipeline for processing `PhiCoordinate`s. It is designed to mirror a principled, human-like approach to understanding and acting on information.

## The Three Phases

### Phase 1: Intent (Love + Wisdom)

**Purpose**: To determine the underlying purpose and guiding principles of the input text.

**Process**: The Intent stage analyzes the Love and Wisdom components of the `PhiCoordinate` to generate a structured `Intent` object, which includes a `purpose` and a list of `guiding_principles`.

### Phase 2: Context (Justice)

**Purpose**: To evaluate the fairness, moral alignment, and situational relevance of the input text.

**Process**: The Context stage uses the Justice component of the `PhiCoordinate` to generate a `QLAEContext` object, which includes a primary domain and a `truth_sense_validation` flag.

### Phase 3: Execution (Power)

**Purpose**: To generate a concrete, actionable plan based on the intent and context.

**Process**: The Execution stage analyzes the Power component of the `PhiCoordinate` to generate an `ExecutionPlan`, which includes a strategy, magnitude, and description.

## Data Flow

The ICE pipeline is the central processing unit of the `TruthSenseTransformer`. The data flow is as follows:

1.  **Input**: The pipeline receives an aligned `PhiCoordinate`.
2.  **Intent**: The Love and Wisdom components are used to generate an `Intent` object.
3.  **Context**: The Justice component is used to generate a `QLAEContext` object.
4.  **Execution**: The Power component is used to generate an `ExecutionPlan`.
5.  **Output**: The `Intent`, `QLAEContext`, and `ExecutionPlan` objects are passed to the `OutputGenerator` to be synthesized into a final, human-readable output.

## Conclusion

The ICE Framework provides a structured and principled approach to semantic analysis. By breaking down the problem into three distinct stages, it allows for a more nuanced and context-aware understanding of meaning than is possible with a single, monolithic model.
