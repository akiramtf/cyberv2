# Performance Comparison Test Guide

## Overview

This script compares the performance of individual models (XGBoost, Random Forest, Neural Network) against the weighted ensemble.

## What It Tests

1. **XGBoost alone** - 40% weight in ensemble
2. **Random Forest alone** - 30% weight in ensemble
3. **Neural Network alone** - 30% weight in ensemble
4. **Weighted Ensemble** (40-30-30) vs. best individual model

## How to Run

```bash
python test_performance_comparison.py
```

## Requirements

- Pre-trained models must exist in `models/trained/`:
  - `xgboost_model.pkl`
  - `random_forest_model.pkl`
  - `neural_network_model.h5`
  - `scaler.pkl`
- Dataset: `dataset2.csv`

If models don't exist, train them first:
```bash
python training/train.py
```

## What the Script Does

### Step 1: Load Dataset
- Loads `dataset2.csv`
- Shows class distribution (legitimate vs. phishing)
- Warns if dataset is imbalanced

### Step 2: Split Dataset (80/20)
- **Critical**: Splits BEFORE testing to avoid data leakage
- Uses stratified split to maintain class proportions
- Uses only TEST SET for evaluation (never training data)

### Step 3: Load Pre-trained Models
- Loads all 3 trained models + scaler
- No re-training occurs (uses existing models)

### Step 4: Extract Features
- Extracts 57 features from each test URL
- Shows progress every 500 URLs
- Takes ~1-2 minutes for 2,000 URLs

### Step 5: Test Individual Models
For each model (XGBoost, RF, NN), calculates:
- **Accuracy**: Overall correctness
- **Precision**: When it predicts phishing, how often is it right?
- **Recall**: Of all phishing URLs, how many did it catch?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Ability to distinguish between classes
- **Confusion Matrix**:
  - True Negatives (TN): Legitimate correctly identified
  - False Positives (FP): Legitimate wrongly flagged as phishing
  - False Negatives (FN): Phishing URLs missed
  - True Positives (TP): Phishing correctly detected

### Step 6: Test Weighted Ensemble
- Combines predictions: `0.4×XGBoost + 0.3×RF + 0.3×NN`
- Calculates same metrics as individual models

### Step 7: Compare All Models
- Ranks models by accuracy
- Identifies best individual model
- Compares ensemble vs. best individual
- Shows improvement percentages

### Step 8: Save Results
- Saves to `performance_comparison_results.csv`

## Output Example

```
================================================================================
  MODEL COMPARISON SUMMARY
================================================================================

Ranking by Accuracy:
----------------------------------------------------------------------------------------------------
Rank   Model                     Accuracy     Precision    Recall       F1-Score     ROC-AUC
----------------------------------------------------------------------------------------------------
🏆 1    Weighted Ensemble         0.9875       0.9900       0.9950       0.9925       0.9850
   2    XGBoost                   0.9850       0.9880       0.9940       0.9910       0.9820
   3    Random Forest             0.9720       0.9750       0.9890       0.9819       0.9650
   4    Neural Network            0.9680       0.9700       0.9900       0.9799       0.9600
----------------------------------------------------------------------------------------------------

================================================================================
  ENSEMBLE vs. BEST INDIVIDUAL MODEL
================================================================================

Best Individual Model: XGBoost
  Accuracy: 0.9850
  F1-Score: 0.9910
  ROC-AUC:  0.9820

Weighted Ensemble:
  Accuracy: 0.9875
  F1-Score: 0.9925
  ROC-AUC:  0.9850

Ensemble Improvement over Best Individual:
  Accuracy: +0.25%
  F1-Score: +0.15%
  ROC-AUC:  +0.31%

✅ Ensemble outperforms best individual model
```

## Understanding the Metrics

### Accuracy
- **What it means**: Percentage of correct predictions
- **Good value**: > 95% for phishing detection
- **Warning**: Can be misleading with imbalanced datasets

### Precision
- **What it means**: When the model says "phishing", how often is it correct?
- **Formula**: TP / (TP + FP)
- **Important for**: Avoiding false alarms (legitimate sites flagged as phishing)

### Recall (Sensitivity)
- **What it means**: Of all real phishing URLs, how many did we catch?
- **Formula**: TP / (TP + FN)
- **Important for**: Security (don't miss phishing attacks)

### F1-Score
- **What it means**: Balance between precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Good value**: > 0.90

### ROC-AUC
- **What it means**: Model's ability to distinguish phishing from legitimate
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Good value**: > 0.90
- **Warning**: 1.0 often indicates data leakage!

### Confusion Matrix

```
                    Predicted
                 Legit  Phishing
Actual  Legit      TN      FP
        Phish      FN      TP
```

- **TN (True Negative)**: Correctly identified legitimate URLs ✅
- **FP (False Positive)**: Legitimate flagged as phishing ❌ (False alarm)
- **FN (False Negative)**: Phishing missed ❌ (Security risk!)
- **TP (True Positive)**: Correctly detected phishing ✅

## Results File

The script saves results to `performance_comparison_results.csv`:

```csv
model,accuracy,precision,recall,f1_score,roc_auc,tp,tn,fp,fn
Weighted Ensemble,0.9875,0.9900,0.9950,0.9925,0.9850,1980,15,5,0
XGBoost,0.9850,0.9880,0.9940,0.9910,0.9820,1976,16,4,4
Random Forest,0.9720,0.9750,0.9890,0.9819,0.9650,1965,12,8,15
Neural Network,0.9680,0.9700,0.9900,0.9799,0.9600,1970,8,12,10
```

## Interpreting Results for Research Report

### For "Results and Discussion" Section

Use this data to:

1. **Show individual model performance**
   ```
   "XGBoost achieved the highest individual accuracy at 98.50% with
   ROC-AUC of 0.9820, followed by Random Forest (97.20%) and Neural
   Network (96.80%)."
   ```

2. **Demonstrate ensemble advantage**
   ```
   "The weighted ensemble (40-30-30) improved accuracy to 98.75%,
   representing a 0.25% improvement over the best individual model."
   ```

3. **Discuss false positives/negatives**
   ```
   "The system achieved 5 false positives and 0 false negatives on
   the 2,000-URL test set, demonstrating strong security with minimal
   user disruption."
   ```

4. **Compare against baselines**
   ```
   "Individual models achieved 96.8-98.5% accuracy, while ensemble
   reached 98.75%, outperforming traditional single-model approaches."
   ```

## Important Notes

### ⚠️ Dataset Imbalance Warning

If you see this warning:
```
⚠️  WARNING: Only 140 legitimate URLs - results may be biased!
```

Your results are likely **invalid** due to severe class imbalance. See main README for solutions.

### ✅ Valid Results Indicators

- ROC-AUC between 0.85 - 0.98 (not 1.0)
- False positives and false negatives both present
- Ensemble slightly better than individuals (not dramatically)
- Reasonable confusion matrix values

### ❌ Invalid Results Indicators (Data Leakage)

- ROC-AUC = 1.0000 (perfect score)
- Zero false negatives or zero false positives
- Accuracy > 99.9%
- All models perform identically

If you see invalid indicators, the test data may have been used for training!

## Troubleshooting

### Error: "Models not found"
```bash
# Train models first
python training/train.py
```

### Error: "Dataset not found"
```bash
# Ensure dataset2.csv exists
ls -lh dataset2.csv
```

### Feature extraction too slow
- Default: Tests 2,000 URLs (~1-2 minutes)
- Edit `SAMPLE_SIZE` in script to reduce (line 251)

### Memory issues
- Reduce `SAMPLE_SIZE` to 1,000 or 500
- Close other applications

## Next Steps

After running this test:

1. **Save the output** for your research report
2. **Copy the metrics** to your Results section
3. **Analyze the confusion matrix** to discuss error types
4. **Compare with literature** (other phishing detection papers)
5. **Create visualizations** (ROC curves, bar charts) if needed

## For Research Paper

This test provides data for:
- **Table 1**: Individual model comparison
- **Table 2**: Ensemble vs. best individual
- **Figure 1**: Performance comparison bar chart
- **Discussion**: Why ensemble works, error analysis
