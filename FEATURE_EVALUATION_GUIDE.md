# Feature-Specific Model Evaluation Guide

This guide explains how to evaluate and train models using different feature subsets to understand the contribution of lexical vs host-based features.

## 📊 Feature Breakdown

Your phishing detector uses **57 total features**:

- **39 Lexical Features**: URL structure, patterns, entropy, special characters
  - Fast to extract (no network calls)
  - Always available
  
- **18 Host-Based Features**: DNS records, SSL certificates, port analysis
  - Slow to extract (requires network lookups)
  - May fail for unreachable domains

## 🎯 Quick Start

### Option 1: Compare All Three Models (Recommended)

```bash
# Fast comparison (lexical features only, host features use defaults)
python evaluate_comparison.py dataset4.csv

# Accurate comparison (with DNS/SSL lookups - SLOW but accurate)
python evaluate_comparison.py dataset4.csv --enable-dns
```

**Output**: Side-by-side comparison of:
1. Lexical-only model (39 features)
2. Host-only model (18 features)
3. Combined model (57 features)

Results saved to: `feature_comparison_results.csv`

---

### Option 2: Evaluate Individual Feature Sets

#### Lexical Features Only (Fast)
```bash
python evaluate_lexical_only.py dataset4.csv
```
- Uses only 39 lexical features
- Fast execution (no network calls)
- Results saved to: `misclassified_lexical_only.csv`

#### Host-Based Features Only (Slow)
```bash
python evaluate_host_only.py dataset4.csv
```
- Uses only 18 host-based features
- ⚠️ **SLOW**: Requires DNS/SSL lookups for each URL
- Results saved to: `misclassified_host_only.csv`

---

## 🏋️ Training Feature-Specific Models

### Train Lexical-Only Model
```bash
python training/train_lexical_only.py --data dataset4.csv

# Limit dataset size for faster training
python training/train_lexical_only.py --data dataset4.csv --max-rows 1000
```
- Trains on 39 lexical features only
- Fast training (no DNS/SSL lookups)
- Saves to: `models/lexical_only/`

### Train Host-Only Model
```bash
python training/train_host_only.py --data dataset4.csv

# Recommended: Use smaller dataset due to slow DNS/SSL lookups
python training/train_host_only.py --data dataset4.csv --max-rows 500
```
- Trains on 18 host-based features only
- ⚠️ **VERY SLOW**: DNS/SSL lookups for each URL
- Saves to: `models/host_only/`

---

## 📈 Understanding Results

### Comparison Output Example

```
Model                Features   Accuracy  Precision    Recall        F1   ROC-AUC       FPR
--------------------------------------------------------------------------------------------
Lexical-Only               39     0.8500     0.8200    0.8800    0.8490    0.9100    0.1500
Host-Based-Only            18     0.7200     0.7000    0.7500    0.7240    0.8000    0.2800
Combined                   57     0.9200     0.9000    0.9400    0.9190    0.9600    0.0800
```

**Key Metrics**:
- **Accuracy**: Overall correctness
- **Precision**: How many predicted phishing sites are actually phishing
- **Recall**: How many actual phishing sites were detected
- **F1**: Harmonic mean of precision and recall
- **FPR** (False Positive Rate): How many safe sites were wrongly blocked (LOWER IS BETTER)

### Expected Findings

1. **Lexical-Only**: 
   - Fast and reliable
   - Good baseline performance
   - Works even when DNS/SSL fails

2. **Host-Only**:
   - Slower but provides complementary signals
   - May perform worse alone
   - Useful for detecting sophisticated attacks

3. **Combined**:
   - Best overall performance
   - Leverages strengths of both feature types
   - Recommended for production use

---

## 🔧 Advanced Options

### Custom Test Split
```bash
# Use 30% for testing instead of default 20%
python evaluate_comparison.py dataset4.csv --test-size 0.3
```

### Custom Output Directory
```bash
python training/train_lexical_only.py --data dataset4.csv --output models/my_lexical_model
```

---

## 📝 Output Files

| File | Description |
|------|-------------|
| `feature_comparison_results.csv` | Comparison metrics for all three models |
| `misclassified_lexical_only.csv` | URLs misclassified by lexical-only model |
| `misclassified_host_only.csv` | URLs misclassified by host-only model |

---

## ⚡ Performance Tips

1. **For Quick Analysis**: Use `evaluate_comparison.py` without `--enable-dns`
   - Host features will use default values
   - Fast execution
   - Good for understanding lexical feature contribution

2. **For Accurate Analysis**: Use `evaluate_comparison.py --enable-dns`
   - Accurate host-based features
   - Slow execution (2-5 seconds per URL)
   - Recommended for final evaluation

3. **For Training**: Start with small datasets
   - Lexical-only: Can handle large datasets (10k+ URLs)
   - Host-only: Limit to 500-1000 URLs due to network lookups

---

## 🎓 Use Cases

### Research Question: "How much do host-based features contribute?"

```bash
# Step 1: Compare all three
python evaluate_comparison.py dataset4.csv --enable-dns

# Step 2: Analyze the results
# - Compare accuracy between Lexical-Only and Combined
# - The difference shows host-based feature contribution
```

### Research Question: "Can we detect phishing without network lookups?"

```bash
# Evaluate lexical-only model
python evaluate_lexical_only.py dataset4.csv

# If accuracy is acceptable, you can deploy without DNS/SSL
# This makes detection faster and more reliable
```

### Research Question: "Which individual features matter most?"

```bash
# Run comparison and check "TOP 5 FEATURES BY MODEL" section
python evaluate_comparison.py dataset4.csv
```

---

## 🐛 Troubleshooting

**Issue**: Host-based evaluation is too slow

**Solution**: 
- Use smaller dataset with `--max-rows`
- Or use `evaluate_comparison.py` without `--enable-dns` for quick analysis

**Issue**: "Missing features" warning

**Solution**: 
- This is normal when DNS/SSL lookups are disabled
- Host features will have default values (0.0 or 1.0)

**Issue**: Low accuracy for host-only model

**Solution**:
- Ensure `--enable-dns` is used for accurate host features
- Host-only models typically perform worse than combined models
- This is expected and shows why combining features is important
