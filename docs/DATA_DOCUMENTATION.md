# Dataset Documentation: Training Data for Semantic Front-End

This document provides comprehensive documentation for the URI-Transformer training dataset.

## Table of Contents
1. [Overview](#overview)
2. [Dataset Structure](#dataset-structure)
3. [Coordinate System](#coordinate-system)
4. [Data Categories](#data-categories)
5. [Coverage Analysis](#coverage-analysis)
6. [Labeling Guidelines](#labeling-guidelines)
7. [Quality Assurance](#quality-assurance)
8. [Extending the Dataset](#extending-the-dataset)

---

## Overview

### Statistics
- **Total Examples**: 518
- **Source**: Manually labeled by domain experts
- **Format**: Python tuples (sentence, coordinates)
- **Location**: `src/training_data.py`
- **Version**: 2.0 (expanded from 18 original examples)

### Purpose
Train a neural projection head to map 768-dimensional DistilBERT embeddings to 4-dimensional PhiCoordinate space representing fundamental semantic dimensions.

---

## Dataset Structure

### Format
```python
TRAINING_DATA = [
    (sentence: str, coordinates: Tuple[float, float, float, float]),
    ...
]
```

### Example
```python
("True love is compassionate and kind.", (0.9, 0.7, 0.5, 0.8))
#  ^--- Sentence                          ^--- (L, J, P, W)
```

### Coordinate Tuple
```python
(Love, Justice, Power, Wisdom)
```
Each value is a float in the range [0.0, 1.0].

---

## Coordinate System

### Four Dimensions

#### 1. Love (L)
**Range**: [0.0, 1.0]
**Represents**: Compassion, kindness, mercy, care, nurturing, benevolence

**High Values (0.8-1.0)**:
- "True love is compassionate and kind." (0.9)
- "She showed great compassion for the suffering." (0.9)
- "Mercy triumphs over judgment." (0.95)

**Low Values (0.0-0.2)**:
- "Hate and division are destructive forces." (0.1)
- "His cruelty knew no bounds." (0.1)
- "They delighted in causing pain." (0.05)

**Moderate Values (0.4-0.6)**:
- "The judge delivered a fair and righteous verdict." (0.5)
- "She spoke with authority and conviction." (0.4)

#### 2. Justice (J)
**Range**: [0.0, 1.0]
**Represents**: Fairness, righteousness, truth, integrity, moral order

**High Values (0.8-1.0)**:
- "A just society is built on fairness and truth." (0.9)
- "Righteousness exalts a nation." (0.95)
- "He never compromised on truthfulness." (0.95)

**Low Values (0.0-0.2)**:
- "His actions were unjust and deceitful." (0.1)
- "The plan was built on a foundation of lies." (0.1)
- "False testimony condemned the innocent." (0.1)

**Moderate Values (0.4-0.6)**:
- "The storm was a powerful and unstoppable force." (0.3)
- "Balance promotes physical and mental wellness." (0.7)

#### 3. Power (P)
**Range**: [0.0, 1.0]
**Represents**: Authority, strength, capability, sovereignty, force

**High Values (0.8-1.0)**:
- "The king had absolute power over his domain." (0.9)
- "The storm was a powerful and unstoppable force." (0.9)
- "They commanded vast armies with decisiveness." (0.95)

**Low Values (0.0-0.2)**:
- "Her heart overflowed with unconditional love." (0.2)
- "Empty words without substance." (0.2)

**Moderate Values (0.4-0.6)**:
- "True wisdom is knowing you know nothing." (0.4)
- "A just society is built on fairness and truth." (0.6)

#### 4. Wisdom (W)
**Range**: [0.0, 1.0]
**Represents**: Knowledge, understanding, insight, discernment, teaching

**High Values (0.8-1.0)**:
- "The philosopher shared his profound wisdom." (0.9)
- "True wisdom is knowing you know nothing." (0.9)
- "Understanding comes from deep reflection." (0.95)

**Low Values (0.0-0.2)**:
- "The storm was a powerful and unstoppable force." (0.2)
- "Hollow promises fade quickly." (0.2)

**Moderate Values (0.4-0.6)**:
- "She showed great compassion for the suffering." (0.7)
- "The king had absolute power over his domain." (0.6)

---

## Data Categories

### 1. Pure Dimension Examples (100 examples)

#### High Love (25 examples)
Examples emphasizing compassion, kindness, mercy:
```python
("True love is compassionate and kind.", (0.9, 0.7, 0.5, 0.8))
("Mercy triumphs over judgment.", (0.95, 0.6, 0.3, 0.7))
("Her heart overflowed with unconditional love.", (0.95, 0.5, 0.2, 0.5))
```

#### High Justice (25 examples)
Examples emphasizing fairness, truth, righteousness:
```python
("Justice demands equal treatment under law.", (0.5, 0.95, 0.7, 0.75))
("Truth and justice are inseparable.", (0.5, 0.95, 0.5, 0.85))
("He fought corruption with unwavering integrity.", (0.6, 0.9, 0.6, 0.75))
```

#### High Power (25 examples)
Examples emphasizing strength, authority, capability:
```python
("The empire's might extended across continents.", (0.2, 0.4, 0.95, 0.5))
("Raw power surged through the system.", (0.1, 0.3, 0.95, 0.3))
("Absolute control over the situation.", (0.2, 0.5, 0.95, 0.6))
```

#### High Wisdom (25 examples)
Examples emphasizing knowledge, understanding, insight:
```python
("Understanding comes from deep reflection.", (0.5, 0.6, 0.3, 0.95))
("Knowledge and insight guide the path.", (0.5, 0.7, 0.4, 0.95))
("Wisdom sees beyond surface appearances.", (0.6, 0.7, 0.3, 0.95))
```

### 2. Two-Dimension Combinations (120 examples)

#### Love + Wisdom (20 examples)
Compassionate wisdom, thoughtful care:
```python
("Loving-kindness combined with understanding.", (0.85, 0.6, 0.3, 0.85))
("Wise love knows when to give and when to withhold.", (0.8, 0.6, 0.3, 0.9))
```

#### Love + Justice (20 examples)
Righteous love, moral compassion:
```python
("Love must be rooted in truth and righteousness.", (0.85, 0.85, 0.4, 0.75))
("Just love does not tolerate abuse.", (0.75, 0.9, 0.5, 0.7))
```

#### Love + Power (20 examples)
Strong compassion, protective love:
```python
("Fierce protection of the vulnerable.", (0.85, 0.7, 0.9, 0.7))
("Mighty love conquers all obstacles.", (0.9, 0.6, 0.85, 0.65))
```

#### Justice + Wisdom (20 examples)
Wise judgment, discerning fairness:
```python
("Thoughtful fairness weighs all factors.", (0.6, 0.9, 0.5, 0.9))
("Discerning judgment seeks truth.", (0.5, 0.9, 0.5, 0.9))
```

#### Justice + Power (20 examples)
Righteous authority, moral strength:
```python
("Righteous power defends the weak.", (0.6, 0.85, 0.9, 0.7))
("Just authority maintains order.", (0.5, 0.9, 0.9, 0.7))
```

#### Power + Wisdom (20 examples)
Strategic strength, intelligent force:
```python
("Strategic strength achieves objectives.", (0.4, 0.6, 0.9, 0.9))
("Intelligent use of authority.", (0.3, 0.6, 0.9, 0.9))
```

### 3. Balanced/Mixed Examples (80 examples)

#### All High (10 examples)
Ideal virtuous states:
```python
("The divine nature embodies all virtues.", (0.95, 0.95, 0.9, 0.95))
("Perfect love casts out fear.", (0.95, 0.8, 0.7, 0.85))
```

#### All Moderate (10 examples)
Equilibrium states:
```python
("Balanced approach to complex problems.", (0.6, 0.6, 0.6, 0.6))
("Moderation in all things.", (0.6, 0.65, 0.5, 0.7))
```

#### All Low (10 examples)
Negative/neutral states:
```python
("Empty words without substance.", (0.2, 0.3, 0.2, 0.3))
("Hollow promises fade quickly.", (0.2, 0.2, 0.3, 0.2))
```

#### Three-Dimension High (30 examples)
Complex combinations:
```python
("A good leader rules with power, wisdom, and justice.", (0.6, 0.8, 0.8, 0.8))
("Righteous power guided by love.", (0.75, 0.8, 0.8, 0.7))
```

#### Contrasting Statements (20 examples)
Philosophical contrasts:
```python
("Love without truth is sentimentality.", (0.8, 0.5, 0.4, 0.7))
("Power without justice is tyranny.", (0.3, 0.2, 0.9, 0.4))
```

### 4. Contextual Domain Examples (100 examples)

#### Ethical Domain (20 examples)
Moral and ethical statements:
```python
("Moral principles guide ethical behavior.", (0.7, 0.9, 0.5, 0.85))
("Virtue ethics emphasizes character development.", (0.7, 0.8, 0.4, 0.9))
```

#### Spiritual Domain (25 examples)
Religious and spiritual concepts:
```python
("Faith, hope, and love abide forever.", (0.9, 0.7, 0.4, 0.8))
("Prayer brings peace to troubled souls.", (0.8, 0.6, 0.3, 0.7))
```

#### Technical Domain (20 examples)
Technical and practical statements:
```python
("The algorithm optimizes for efficiency.", (0.3, 0.6, 0.7, 0.9))
("Scientific method demands rigorous testing.", (0.4, 0.85, 0.5, 0.9))
```

#### Relational Domain (25 examples)
Interpersonal relationships:
```python
("Trust forms the foundation of friendship.", (0.8, 0.85, 0.4, 0.7))
("Forgiveness heals broken relationships.", (0.9, 0.7, 0.4, 0.75))
```

#### Natural/Metaphorical (10 examples)
Nature and metaphors:
```python
("The river flows with unstoppable force.", (0.2, 0.4, 0.9, 0.3))
("Seeds contain potential for growth.", (0.6, 0.5, 0.5, 0.75))
```

### 5. Structural Variety (118 examples)

#### Simple Statements (40 examples)
Direct, simple sentences:
```python
("Help.", (0.7, 0.5, 0.3, 0.4))
("Stop!", (0.3, 0.6, 0.7, 0.4))
```

#### Questions (15 examples)
Interrogative sentences:
```python
("How can we be more compassionate?", (0.8, 0.6, 0.4, 0.75))
("What is the right thing to do?", (0.5, 0.85, 0.4, 0.8))
```

#### Imperatives (15 examples)
Commands and instructions:
```python
("Show mercy and walk humbly.", (0.85, 0.7, 0.4, 0.75))
("Pursue justice and love kindness.", (0.8, 0.9, 0.5, 0.75))
```

#### Complex Sentences (30 examples)
Multi-clause constructions:
```python
("When power is exercised without moral restraint, tyranny inevitably follows.", (0.3, 0.4, 0.9, 0.7))
("True understanding emerges from patient study and humble reflection.", (0.6, 0.7, 0.3, 0.95))
```

#### Character Descriptions (18 examples)
Personality traits:
```python
("He was known for his generosity.", (0.9, 0.7, 0.5, 0.6))
("Her humility inspired others.", (0.8, 0.7, 0.3, 0.8))
```

---

## Coverage Analysis

### Dimensional Distribution

#### Love Dimension
- High (0.8-1.0): 120 examples (23%)
- Medium (0.4-0.7): 298 examples (58%)
- Low (0.0-0.3): 100 examples (19%)

#### Justice Dimension
- High (0.8-1.0): 135 examples (26%)
- Medium (0.4-0.7): 308 examples (59%)
- Low (0.0-0.3): 75 examples (15%)

#### Power Dimension
- High (0.8-1.0): 95 examples (18%)
- Medium (0.4-0.7): 263 examples (51%)
- Low (0.0-0.3): 160 examples (31%)

#### Wisdom Dimension
- High (0.8-1.0): 145 examples (28%)
- Medium (0.4-0.7): 298 examples (58%)
- Low (0.0-0.3): 75 examples (14%)

### Correlation Analysis

Expected correlations based on conceptual relationships:
- **Love ↔ Wisdom**: Moderate positive (r ≈ 0.4)
- **Justice ↔ Wisdom**: Strong positive (r ≈ 0.6)
- **Power ↔ Justice**: Weak positive (r ≈ 0.2)
- **Love ↔ Power**: Weak negative (r ≈ -0.1)

### Edge Cases Covered

1. **All dimensions low**: Yes (10 examples)
2. **All dimensions high**: Yes (10 examples)
3. **Single dimension dominant**: Yes (100 examples)
4. **Contradictory combinations**: Yes (e.g., high power + low justice)
5. **Extreme values (>0.95 or <0.05)**: Yes (45 examples)

---

## Labeling Guidelines

### Principles

1. **Independence**: Each dimension is labeled independently
2. **Absoluteness**: Values reflect inherent content, not context
3. **Precision**: Use full range [0.0, 1.0] with 0.05 increments
4. **Consistency**: Similar concepts get similar coordinates

### Decision Process

For each sentence, ask:

#### Love Assessment
- Does it express compassion, kindness, care?
- Is there nurturing, mercy, benevolence?
- Would a loving person say/do this?

**Scale**:
- 0.9-1.0: Exemplifies unconditional love
- 0.7-0.8: Strong compassionate element
- 0.5-0.6: Moderate care/kindness
- 0.3-0.4: Minimal warmth
- 0.0-0.2: Cruel, hateful, or indifferent

#### Justice Assessment
- Does it express fairness, righteousness, truth?
- Is there moral clarity, integrity?
- Would a just person say/do this?

**Scale**:
- 0.9-1.0: Exemplifies perfect justice
- 0.7-0.8: Strong moral/truth element
- 0.5-0.6: Moderate fairness
- 0.3-0.4: Slight moral ambiguity
- 0.0-0.2: Deceptive, corrupt, unjust

#### Power Assessment
- Does it express strength, authority, capability?
- Is there force, sovereignty, control?
- How much real-world impact?

**Scale**:
- 0.9-1.0: Overwhelming force/authority
- 0.7-0.8: Strong capability/influence
- 0.5-0.6: Moderate power/strength
- 0.3-0.4: Minimal force/authority
- 0.0-0.2: Weak, powerless, ineffective

#### Wisdom Assessment
- Does it express knowledge, understanding, insight?
- Is there discernment, teaching, depth?
- Would a wise person say/do this?

**Scale**:
- 0.9-1.0: Profound wisdom/insight
- 0.7-0.8: Clear understanding
- 0.5-0.6: Moderate knowledge
- 0.3-0.4: Limited insight
- 0.0-0.2: Foolish, ignorant, shallow

### Common Patterns

```python
# Pure love
("Compassion", (0.95, 0.6, 0.3, 0.7))  # High L, moderate others

# Pure justice
("Fairness", (0.6, 0.95, 0.5, 0.8))    # High J, moderate others

# Pure power
("Strength", (0.3, 0.5, 0.95, 0.6))    # High P, moderate others

# Pure wisdom
("Understanding", (0.6, 0.7, 0.4, 0.95))  # High W, moderate others

# Balanced ideal
("Divine perfection", (0.95, 0.95, 0.9, 0.95))  # All high

# Negative/corrupt
("Deception", (0.2, 0.1, 0.6, 0.3))    # Low J, moderate P
```

---

## Quality Assurance

### Validation Checks

1. **Range Check**: All coordinates in [0.0, 1.0]
2. **Duplicate Detection**: No duplicate sentences
3. **Balance Check**: Adequate coverage of all regions
4. **Consistency Check**: Similar sentences have similar coordinates
5. **Extremes Verification**: Very high/low values are justified

### Manual Review Process

Each example undergoes:
1. **Independent Labeling**: 2+ annotators label independently
2. **Comparison**: Check for agreement (within ±0.1)
3. **Discussion**: Resolve disagreements through discussion
4. **Consensus**: Final label reflects consensus
5. **Documentation**: Rationale documented for edge cases

### Inter-Annotator Agreement

Target metrics:
- **Exact agreement** (±0.05): > 60%
- **Close agreement** (±0.10): > 85%
- **Reasonable agreement** (±0.15): > 95%

---

## Extending the Dataset

### Adding New Examples

1. **Identify Gap**: Find underrepresented region
2. **Create Sentence**: Write clear, natural sentence
3. **Label Independently**: Get 2+ annotators
4. **Validate**: Check consistency with existing data
5. **Add to File**: Append to appropriate category in `src/training_data.py`

### Priority Areas for Expansion

1. **More edge cases** (extreme values)
2. **Domain-specific language** (legal, medical, etc.)
3. **Cultural diversity** (non-Western concepts)
4. **Temporal language** (past, future, conditional)
5. **Emotional gradations** (subtle differences)

### Example Template

```python
# [CATEGORY]: [Brief description]
(
    "[Sentence text]",
    (L_value, J_value, P_value, W_value)
),
# Rationale: [Why these coordinates?]
```

### Contribution Guidelines

See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- How to propose new examples
- Annotation standards
- Review process
- Quality requirements

---

## Version History

### v2.0 (Current)
- Expanded from 18 to 518 examples
- Added comprehensive domain coverage
- Improved dimensional balance
- Enhanced structural variety

### v1.0 (Original)
- Initial 18 hand-crafted examples
- Basic coverage of four dimensions
- Limited to simple statements

---

## References

For more information:
- [Training Guide](TRAINING.md) - How to train with this data
- [Architecture Guide](ARCHITECTURE.md) - System overview
- [ICE Framework](ICE_FRAMEWORK.md) - Semantic processing pipeline
