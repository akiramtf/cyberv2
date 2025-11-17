# Testing Guide for Results and Discussion Section

This guide explains how to test the phishing detection system and collect results for your research report.

---

## Quick Start

```bash
# 1. Train the full model
python training/train.py --data dataset2.csv --output models/trained

# 2. Test individual models comparison
python test_individual_models.py

# 3. Run ablation study
python test_ablation_study.py
```

---

## Test Scenarios for Research Report

### 1. **Overall System Performance** (Section 5.1)

**Command:**
```bash
python training/train.py --data dataset2.csv --output models/trained
```

**What to collect:**
- ✅ Accuracy (target: 92-95%)
- ✅ Precision (target: 90-93%)
- ✅ Recall (target: 88-92%)
- ✅ F1-Score (target: 89-92%)
- ✅ ROC-AUC (target: 0.94-0.97)
- ✅ Confusion Matrix (TN, FP, FN, TP)

**Example output for report:**
```
Our ensemble phishing detection system demonstrates strong performance:
- Accuracy: 94.37%
- Precision: 92.13%
- Recall: 89.87%
- F1-Score: 90.98%
- ROC-AUC: 0.9712
```

---

### 2. **Individual Model Performance** (Section 5.2.1)

**Command:**
```bash
python test_individual_models.py
```

**What to collect:**
- XGBoost standalone performance
- Random Forest standalone performance
- Neural Network standalone performance
- Ensemble performance
- Improvement gained from ensembling

**Example output for report:**
```
Individual Model Performance:

XGBoost (40% weight):
  Accuracy:  0.9293
  Precision: 0.9194
  Recall:    0.8856

Random Forest (30% weight):
  Accuracy:  0.9012
  Precision: 0.8845
  Recall:    0.8923

Neural Network (30% weight):
  Accuracy:  0.8934
  Precision: 0.8512
  Recall:    0.9123

Weighted Ensemble (40-30-30):
  Accuracy:  0.9437  ← Best performance
  Precision: 0.9213
  Recall:    0.8987

Ensemble improvement: +1.44% accuracy over best individual model
```

---

### 3. **Feature Importance Analysis** (Section 5.3)

**Command:**
```bash
python training/train.py --data dataset2.csv --output models/trained
# Check the output for "Top 10 Most Important Features"
```

**What to collect:**
- Top 10 feature rankings
- Feature importance scores
- Feature category breakdown

**Example output for report:**
```
Top 10 Most Influential Features:

1. url_length (0.124): Phishing URLs significantly longer on average
2. hostname_entropy (0.108): Random subdomains indicate suspicious behavior
3. num_dots (0.095): Excessive subdomain nesting
4. has_suspicious_tld (0.089): High-risk TLDs (.tk, .ml, .ga)
5. ssl_cert_age_days (0.081): Newly issued certificates flag emerging threats
6. num_subdomains (0.076): Legitimate sites rarely exceed 2-3 subdomains
7. digit_ratio (0.072): High digit concentration suggests randomization
8. path_depth (0.068): Deep directory structures used for obfuscation
9. has_ip_address (0.064): IP-based URLs almost exclusively phishing
10. has_dns_mx_record (0.059): Legitimate businesses have email infrastructure

Feature Category Contribution:
- Lexical features: 60% of total importance
- Network/Host features: 25% of total importance
- SSL/Crypto features: 10% of total importance
- Structural features: 5% of total importance
```

---

### 4. **Ablation Study** (Section 5.4 - Feature Impact)

**Command:**
```bash
python test_ablation_study.py
```

**What to collect:**
- Performance with all features
- Performance with lexical features only
- Performance without SSL features
- Performance without DNS features
- Impact of each feature group

**Example output for report:**
```
Ablation Study Results:

1. All 57 Features (Baseline):
   Accuracy: 0.9437

2. Lexical Features Only (40 features):
   Accuracy: 0.9124 (Δ-0.0313)
   → Removing network features reduces accuracy by 3.13%

3. Without SSL Features (50 features):
   Accuracy: 0.9301 (Δ-0.0136)
   → SSL features contribute 1.36% to accuracy

4. Without DNS Features (49 features):
   Accuracy: 0.9352 (Δ-0.0085)
   → DNS features contribute 0.85% to accuracy

Feature Group Contribution Ranking:
1. SSL Features: 0.0136 accuracy loss when removed
2. DNS Features: 0.0085 accuracy loss when removed
```

---

### 5. **Different Dataset Sizes** (Section 5.X - Scalability)

**Commands:**
```bash
# Small dataset (1k URLs)
python training/train.py --data dataset2.csv --max-rows 1000 --output models/test_1k

# Medium dataset (10k URLs)
python training/train.py --data dataset2.csv --max-rows 10000 --output models/test_10k

# Large dataset (25k URLs)
python training/train.py --data dataset2.csv --max-rows 25000 --output models/test_25k

# Full dataset (49k URLs)
python training/train.py --data dataset2.csv --output models/trained
```

**What to collect:**
- Performance vs dataset size
- Training time vs dataset size
- Model convergence behavior

**Example output for report:**
```
Performance vs Dataset Size:

Dataset Size | Accuracy | Precision | Recall | Training Time
-------------|----------|-----------|--------|---------------
1,000 URLs   | 0.8745   | 0.8512    | 0.8234 | ~30 seconds
10,000 URLs  | 0.9234   | 0.9012    | 0.8845 | ~2 minutes
25,000 URLs  | 0.9389   | 0.9156    | 0.8923 | ~5 minutes
49,208 URLs  | 0.9437   | 0.9213    | 0.8987 | ~10 minutes

Observation: Performance plateaus after ~25k samples, suggesting diminishing returns.
```

---

### 6. **Train/Test Split Sensitivity** (Section 5.X - Methodology Validation)

**Commands:**
```bash
# 90/10 split
python training/train.py --data dataset2.csv --test-size 0.1 --output models/split_90_10

# 80/20 split (default)
python training/train.py --data dataset2.csv --test-size 0.2 --output models/split_80_20

# 70/30 split
python training/train.py --data dataset2.csv --test-size 0.3 --output models/split_70_30
```

**What to collect:**
- Performance consistency across splits
- Variance in metrics

**Example output for report:**
```
Train/Test Split Analysis:

Split Ratio | Test Samples | Accuracy | Std Dev
------------|--------------|----------|--------
90/10       | 4,921        | 0.9456   | ±0.0012
80/20       | 9,842        | 0.9437   | ±0.0008
70/30       | 14,763       | 0.9423   | ±0.0015

Observation: Consistent performance (94.2-94.6%) across different splits validates model stability.
```

---

### 7. **Error Analysis** (Section 5.5)

After training, manually inspect false positives and false negatives:

**Command:**
```bash
# Train model
python training/train.py --data dataset2.csv --output models/trained

# Then use Python to analyze errors:
```

Create `analyze_errors.py`:
```python
import pandas as pd
from src.phishing_detector.detector import PhishingDetector

# Load model
detector = PhishingDetector()
detector.load('models/trained')

# Test URLs
test_urls = [
    "https://dev.staging-app.vercel.app",  # Legitimate but may be flagged
    "http://paypal-verify.tk/login",       # Phishing
    "https://apple.com",                   # Legitimate (whitelisted)
    "http://192.168.1.1/admin",            # IP address
]

print("\nError Analysis:\n" + "="*60)
for url in test_urls:
    result = detector.predict(url)
    print(f"\nURL: {url}")
    print(f"Prediction: {'PHISHING' if result['is_phishing'] else 'SAFE'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Source: {result['prediction_source']}")
```

---

### 8. **Performance Metrics** (Section 5.X - Real-Time Deployment)

**Test inference speed:**

Create `test_performance.py`:
```python
import time
from src.phishing_detector.detector import PhishingDetector

detector = PhishingDetector()
detector.load('models/trained')

test_urls = [
    "https://google.com",
    "http://suspicious-phishing.tk",
    "https://github.com",
] * 100  # 300 URLs

start = time.time()
results = detector.predict_batch(test_urls)
elapsed = time.time() - start

print(f"\nPerformance Metrics:")
print(f"Total URLs: {len(test_urls)}")
print(f"Total time: {elapsed:.2f}s")
print(f"Average time per URL: {(elapsed/len(test_urls))*1000:.2f}ms")
print(f"Throughput: {len(test_urls)/elapsed:.2f} URLs/second")
```

**Example output for report:**
```
Performance Metrics:
- Average response time: 145ms per URL
- Whitelist lookup: ~0.08ms
- Feature extraction: ~95ms
- ML inference: ~12ms
- Throughput: ~6.9 URLs/second (single-threaded)
```

---

## How to Use Results in Your Report

### Section 5.1: Overall System Performance
→ Use results from **Test 1** (main training output)

### Section 5.2: Model Component Analysis
→ Use results from **Test 2** (individual models comparison)

### Section 5.3: Feature Importance Analysis
→ Use results from **Test 3** (feature importance from training)

### Section 5.4: Domain Whitelist Impact
→ Compare inference time with/without whitelist hits

### Section 5.5: Error Analysis
→ Use results from **Test 7** (error analysis script)

### Section 5.6: Comparison with Existing Approaches
→ Use benchmark comparisons from literature

### Section 5.7: Real-World Deployment Considerations
→ Use results from **Test 8** (performance metrics)

### Section 5.8: Discussion of Key Contributions
→ Synthesize findings from all tests

---

## Tips for Writing Results Section

**Good Practice:**
```
Our ensemble system achieved 94.37% accuracy, outperforming individual models:
- XGBoost: 92.93%
- Random Forest: 90.12%
- Neural Network: 89.34%

The ensemble provides a 1.44% improvement over the best individual model,
validating the weighted voting approach.
```

**Include Visualizations:**
- Confusion matrices
- ROC curves
- Feature importance bar charts
- Training loss curves
- Performance vs dataset size plots

**Report Confidence Intervals:**
```
Accuracy: 94.37% ± 0.12% (95% confidence interval across 5 random seeds)
```

---

## Checklist Before Writing Report

- [ ] Trained model on full dataset
- [ ] Collected all 5 core metrics (accuracy, precision, recall, F1, ROC-AUC)
- [ ] Compared individual models vs ensemble
- [ ] Analyzed feature importance
- [ ] Ran ablation study
- [ ] Tested different dataset sizes
- [ ] Measured inference performance
- [ ] Performed error analysis
- [ ] Documented all configurations used

---

## Common Issues

**Issue: Training crashes with memory error**
```bash
# Solution: Use smaller dataset
python training/train.py --data dataset2.csv --max-rows 10000
```

**Issue: Neural network not converging**
```bash
# Solution: Check training logs for early stopping
# May need to retrain with different random seed
```

**Issue: Very low accuracy (<80%)**
```bash
# Check data quality:
# - Ensure labels are correct (0=legitimate, 1=phishing)
# - Check for data corruption
# - Verify feature extraction is working
```

---

Good luck with your Results and Discussion section! 🎉
