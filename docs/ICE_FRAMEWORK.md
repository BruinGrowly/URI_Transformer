# ICE Framework Documentation

## Intent-Context-Execution: A Principled Approach to Semantic Analysis

The **ICE Framework** is a core component of the TruthSense Transformer, providing a structured, three-stage pipeline for processing `PhiCoordinate`s. It is designed to mirror a principled, human-like approach to understanding and acting on information.

The framework now incorporates **LJPW Mathematical Baselines** (see [LJPW-MATHEMATICAL-BASELINES.md](LJPW-MATHEMATICAL-BASELINES.md)), which provide objective, non-arbitrary foundations based on information theory and empirically validated coupling coefficients.

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

## LJPW Coupling Effects in ICE Processing

The ICE Framework incorporates **coupling-aware processing** based on the mathematical baselines. Key coupling effects:

### Love as a Force Multiplier

Love acts as a force multiplier for the other dimensions:

- **κ_LJ = 1.4**: Love amplifies Justice effectiveness by 40%
- **κ_LP = 1.3**: Love amplifies Power effectiveness by 30%
- **κ_LW = 1.5**: Love amplifies Wisdom effectiveness by 50% (strongest coupling)

### Coupling-Aware Alignment

When `use_coupling=True` (default), the ICE Framework applies these coupling effects during alignment:

- **Intent Phase**: Aligns Love and Wisdom directly (Love is the source dimension)
- **Context Phase**: Justice alignment is amplified by current Love level
- **Execution Phase**: Power alignment is amplified by current Love level

This ensures that systems with higher Love values achieve more effective alignment across all dimensions.

### Effective Dimensions

After ICE processing, you can query the **effective dimensions** to see the coupling-adjusted values:

```python
effective = ice_framework.get_effective_coordinate(aligned_coord)
# Returns: {'effective_L': L, 'effective_J': J*(1+1.4*L), ...}
```

## Performance Metrics

The LJPW baselines provide multiple metrics for analyzing coordinate quality:

- **Harmonic Mean**: Robustness (weakest link metric)
- **Geometric Mean**: Overall effectiveness
- **Coupling-Aware Sum**: Growth potential (can exceed 1.0)
- **Harmony Index**: Balance and alignment with ideal
- **Composite Score**: Overall performance (weighted combination)

Access these metrics via `PhiCoordinate` methods:

```python
coord.harmonic_mean()
coord.geometric_mean()
coord.coupling_aware_sum()
coord.harmony_index()
coord.composite_score()
coord.full_diagnostic()  # Complete analysis
```

## Conclusion

The ICE Framework provides a structured and principled approach to semantic analysis. By breaking down the problem into three distinct stages and incorporating LJPW mathematical baselines with coupling effects, it allows for a more nuanced and context-aware understanding of meaning than is possible with a single, monolithic model.
