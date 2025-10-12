# GETTING_STARTED.md

## 🎯 Choose Your Learning Path

We've created multiple ways to learn URI-Transformer based on your learning style and time available.

---

## 🏃‍♂️ **30-Second Demo** - The Fastest Way to See Results

**Perfect if**: You want to see URI-Transformer work immediately

```bash
git clone https://github.com/BruinGrowly/URI_Transformer.git
cd URI_Transformer
pip install -r requirements.txt
python quick_start_demo.py
```

**What you'll see**:
- ✅ Transformer initialization
- ✅ 3 example transformations with results
- ✅ Interactive text input (try your own!)
- ✅ Clear explanation of results

---

## 🎓 **5-Minute Interactive Tutorial** - Learn Step-by-Step

**Perfect if**: You want to understand HOW it works

```bash
python 5_minute_tutorial.py
```

**What you'll learn**:
1. 🧠 ICE Framework (Intent → Context → Execution)
2. 🌍 4D Semantic Coordinate System
3. 🔄 Live transformation examples
4. 🎭 5 Execution Strategies
5. 🎮 Interactive experiments

**Interactive features**:
- Press Enter to progress through steps
- Type your own sentences to transform
- See how different texts map to coordinates

---

## 💻 **Developer Examples** - Practical Copy-Paste Code

**Perfect if**: You want to integrate URI-Transformer into your projects

```bash
python simple_examples.py
```

**7 Practical Examples Covered**:

1. **Basic Transformation** - The most common usage pattern
2. **Batch Processing** - Analyze multiple texts efficiently
3. **Emotional Analysis** - Understand text emotions
4. **Content Filtering** - Safety and alignment checks
5. **Custom Domains** - Specialized processing contexts
6. **Performance Comparison** - ICE vs Standard transformers
7. **Easy Integration** - Wrapper class for existing projects

**Integration Helper Class**:
```python
analyzer = SimpleSemanticAnalyzer()
sentiment = analyzer.analyze_sentiment("Help others learn")
safety = analyzer.get_safety_score("Share knowledge freely")
response_type = analyzer.suggest_response("I need guidance")
```

---

## 📋 **One-Liner Reference** - Ultra-Compact Functions

**Perfect if**: You need quick utilities and reference code

```bash
python one_liner_examples.py
```

**20+ One-Liner Functions**:

```python
# Basic usage
result = basic_transform("Help others")

# Quick analysis
sentiment = get_sentiment("Be kind")              # "LOVE"
safe = is_safe("Share knowledge")                 # True
response = suggest_response("Need help")          # "compassionate_action"

# Coordinate extraction
coords = get_coordinates("Stay strong")           # (0.2, 0.9, 0.3, 0.4)
love_score = love_score("Show compassion")        # 0.95

# Batch operations
results = batch_analyze(["Be kind", "Stay strong"])
safe_texts = filter_safe_texts(all_texts)
sorted_by_love = sort_by_love(texts)

# Classification
classification = simple_classify("Learn daily")   # "WISDOM (0.87)"
summary = text_summary("Act with justice")        # Full analysis dict
```

---

## 📖 **Comprehensive Guide** - Complete Documentation

**Perfect if**: You want to understand everything

**Read**: `QUICK_START.md` or this repository's main `README.md`

**Topics covered**:
- 🧠 Deep dive into ICE Framework
- 📊 Performance metrics and benchmarks  
- 🔧 Architecture details
- 🌐 Integration with LLMs
- 🧪 Advanced testing strategies
- 🤝 Contributing guidelines

---

## 🎯 **Recommendations by Use Case**

### 👶 **First-Time User**
```bash
python quick_start_demo.py  # 30 seconds
```

### 🔧 **Developer Adding to Project**
```bash
python simple_examples.py   # 3 minutes, copy-paste ready
```

### 🎓 **Student/Researcher**
```bash
python 5_minute_tutorial.py # 5 minutes, interactive learning
```

### ⚡ **Experienced Developer**
```bash
python one_liner_examples.py # 1 minute, reference functions
```

### 📚 **Complete Understanding**
Read `README.md` + `QUICK_START.md` + run one tutorial

---

## 🚀 **Your First URI-Transformer Code**

**Copy-paste this to get started immediately**:

```python
from src.ice_uri_transformer import ICEURITransformer

# Initialize
transformer = ICEURITransformer()

# Transform text
result = transformer.transform(
    "Help others with compassion",
    thought_type="moral_judgment",
    context_domain="ethical"
)

# See results
print(f"Intent: {result.intent_coordinates}")  # (LOVE, POWER, WISDOM, JUSTICE)
print(f"Strategy: {result.execution_strategy}")  # How to respond
print(f"Integrity: {result.semantic_integrity:.2%}")  # Meaning preserved
print(f"Alignment: {result.divine_alignment:.3f}")  # Value alignment
```

**That's it! You're now using URI-Transformer! 🎉**

---

## 🆘 **Need Help?**

1. **🏃‍♂️ Quick Demo**: `python quick_start_demo.py`
2. **🎓 Tutorial**: `python 5_minute_tutorial.py`  
3. **💻 Examples**: `python simple_examples.py`
4. **📋 Reference**: `python one_liner_examples.py`
5. **📚 Docs**: Read `README.md` and `QUICK_START.md`
6. **🐛 Issues**: [GitHub Issues](https://github.com/BruinGrowly/URI_Transformer/issues)

**Choose the path that works best for you! 🌟**