#!/usr/bin/env python3
"""
Performance Comparison Test Script
Tests individual models vs. weighted ensemble using pre-trained models.

This script:
1. Loads pre-trained models from models/trained/
2. Splits dataset properly (80/20) to avoid data leakage
3. Tests XGBoost, Random Forest, Neural Network individually
4. Tests weighted ensemble (40-30-30 weights)
5. Compares ensemble vs. best individual model
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from phishing_detector.detector import PhishingDetector


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}\n")


def print_metrics(model_name, y_true, y_pred, y_pred_proba):
    """Print comprehensive metrics for a model"""
    print(f"\n{model_name} Performance:")
    print("-" * 60)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC (handle edge cases)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        roc_auc = 0.0

    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives  (TN): {tn:6d}  (Legitimate correctly identified)")
    print(f"    False Positives (FP): {fp:6d}  (Legitimate wrongly flagged)")
    print(f"    False Negatives (FN): {fn:6d}  (Phishing missed)")
    print(f"    True Positives  (TP): {tp:6d}  (Phishing correctly detected)")

    # Calculate rates
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
        fpr = fp / (tn + fp)
        print(f"\n  Specificity (True Negative Rate): {specificity:.4f}")
        print(f"  False Positive Rate:              {fpr:.4f}")

    if (tp + fn) > 0:
        tpr = tp / (tp + fn)
        fnr = fn / (tp + fn)
        print(f"  True Positive Rate (Recall):      {tpr:.4f}")
        print(f"  False Negative Rate:              {fnr:.4f}")

    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def test_individual_model(model, X_test, y_test, model_name):
    """Test an individual model and return predictions"""
    print(f"\n[Testing {model_name}...]")

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Print metrics
    metrics = print_metrics(model_name, y_test, y_pred, y_pred_proba)

    return metrics, y_pred_proba


def test_ensemble(detector, X_test, y_test, individual_probas):
    """Test weighted ensemble (40-30-30)"""
    print(f"\n[Testing Weighted Ensemble (XGB:40%, RF:30%, NN:30%)]")

    # Calculate weighted ensemble
    xgb_proba = individual_probas['xgboost']
    rf_proba = individual_probas['random_forest']
    nn_proba = individual_probas['neural_network']

    # Apply weights: XGBoost 40%, RF 30%, NN 30%
    ensemble_proba = (0.4 * xgb_proba + 0.3 * rf_proba + 0.3 * nn_proba)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    # Print metrics
    metrics = print_metrics("Weighted Ensemble", y_test, ensemble_pred, ensemble_proba)

    return metrics


def compare_models(results):
    """Compare all models and highlight the best"""
    print_header("MODEL COMPARISON SUMMARY")

    # Create comparison table
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('accuracy', ascending=False)

    print("Ranking by Accuracy:")
    print("-" * 100)
    print(f"{'Rank':<6} {'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 100)

    for idx, row in df_results.iterrows():
        rank = df_results.index.get_loc(idx) + 1
        marker = "🏆" if rank == 1 else "  "
        print(f"{marker} {rank:<4} {row['model']:<25} {row['accuracy']:<12.4f} {row['precision']:<12.4f} "
              f"{row['recall']:<12.4f} {row['f1_score']:<12.4f} {row['roc_auc']:<12.4f}")

    print("-" * 100)

    # Find best individual model
    individual_models = df_results[df_results['model'] != 'Weighted Ensemble']
    ensemble_row = df_results[df_results['model'] == 'Weighted Ensemble'].iloc[0]
    best_individual = individual_models.iloc[0]

    print_header("ENSEMBLE vs. BEST INDIVIDUAL MODEL")

    print(f"Best Individual Model: {best_individual['model']}")
    print(f"  Accuracy: {best_individual['accuracy']:.4f}")
    print(f"  F1-Score: {best_individual['f1_score']:.4f}")
    print(f"  ROC-AUC:  {best_individual['roc_auc']:.4f}")

    print(f"\nWeighted Ensemble:")
    print(f"  Accuracy: {ensemble_row['accuracy']:.4f}")
    print(f"  F1-Score: {ensemble_row['f1_score']:.4f}")
    print(f"  ROC-AUC:  {ensemble_row['roc_auc']:.4f}")

    # Calculate improvement
    acc_improvement = ((ensemble_row['accuracy'] - best_individual['accuracy']) / best_individual['accuracy']) * 100
    f1_improvement = ((ensemble_row['f1_score'] - best_individual['f1_score']) / best_individual['f1_score']) * 100
    auc_improvement = ((ensemble_row['roc_auc'] - best_individual['roc_auc']) / best_individual['roc_auc']) * 100

    print(f"\nEnsemble Improvement over Best Individual:")
    print(f"  Accuracy: {acc_improvement:+.2f}%")
    print(f"  F1-Score: {f1_improvement:+.2f}%")
    print(f"  ROC-AUC:  {auc_improvement:+.2f}%")

    if acc_improvement > 0:
        print(f"\n✅ Ensemble outperforms best individual model")
    elif acc_improvement == 0:
        print(f"\n➡️  Ensemble equals best individual model")
    else:
        print(f"\n⚠️  Best individual model outperforms ensemble")

    return df_results


def main():
    print_header("PHISHING DETECTION: PERFORMANCE COMPARISON TEST")
    print("Testing pre-trained models from: models/trained/")
    print("Dataset: dataset2.csv")
    print("Test Strategy: 80/20 train-test split with stratification")

    # Configuration
    MODEL_PATH = 'models/trained'
    DATASET_PATH = 'dataset2.csv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    SAMPLE_SIZE = 2000  # Test on 2000 URLs for speed

    # Step 1: Load full dataset
    print_header("STEP 1: Loading Dataset")
    print(f"Loading {DATASET_PATH}...")

    df = pd.read_csv(DATASET_PATH, on_bad_lines='skip')
    print(f"Total rows loaded: {len(df)}")

    # Check required columns
    if 'url' not in df.columns or 'label' not in df.columns:
        print("❌ Error: Dataset must have 'url' and 'label' columns")
        return

    # Remove invalid rows
    df = df.dropna(subset=['url', 'label'])
    df = df[df['url'].str.strip() != '']
    print(f"Valid rows: {len(df)}")

    # Show class distribution
    legitimate_count = len(df[df['label'] == 0])
    phishing_count = len(df[df['label'] == 1])
    print(f"\nClass Distribution:")
    print(f"  Legitimate (label=0): {legitimate_count:,} ({legitimate_count/len(df)*100:.2f}%)")
    print(f"  Phishing (label=1):   {phishing_count:,} ({phishing_count/len(df)*100:.2f}%)")

    if legitimate_count < 100:
        print(f"\n⚠️  WARNING: Only {legitimate_count} legitimate URLs - results may be biased!")

    # Step 2: Split dataset FIRST (critical to avoid data leakage)
    print_header("STEP 2: Splitting Dataset")
    print(f"Splitting: {(1-TEST_SIZE)*100:.0f}% train / {TEST_SIZE*100:.0f}% test (stratified)")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['label']
    )

    print(f"Training set: {len(train_df):,} URLs")
    print(f"Test set:     {len(test_df):,} URLs")

    # Sample from test set for faster processing
    if len(test_df) > SAMPLE_SIZE:
        test_sample = test_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        print(f"\nUsing random sample: {SAMPLE_SIZE:,} URLs from test set")
    else:
        test_sample = test_df
        print(f"\nUsing full test set: {len(test_sample):,} URLs")

    test_urls = test_sample['url'].tolist()
    y_test = test_sample['label'].values

    # Step 3: Load pre-trained models
    print_header("STEP 3: Loading Pre-trained Models")
    print(f"Loading models from: {MODEL_PATH}/")

    detector = PhishingDetector(enable_dns_lookup=False, random_state=RANDOM_STATE)

    try:
        detector.load(MODEL_PATH)
        print("✅ Models loaded successfully")
        print(f"   - XGBoost model: {MODEL_PATH}/xgboost_model.pkl")
        print(f"   - Random Forest model: {MODEL_PATH}/random_forest_model.pkl")
        print(f"   - Neural Network model: {MODEL_PATH}/neural_network_model.h5")
        print(f"   - Feature scaler: {MODEL_PATH}/scaler.pkl")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nPlease ensure models are trained first:")
        print("  python training/train.py")
        return

    # Step 4: Extract features from test set
    print_header("STEP 4: Extracting Features from Test Set")
    print(f"Processing {len(test_urls):,} URLs...")
    print("(This may take 1-2 minutes)\n")

    features_list = []
    valid_indices = []

    for i, url in enumerate(test_urls):
        if (i + 1) % 500 == 0:
            print(f"  Processed: {i+1}/{len(test_urls)} URLs ({(i+1)/len(test_urls)*100:.1f}%)")

        try:
            features = detector.feature_extractor.extract_features(url)
            if features is not None:
                features_list.append(features)
                valid_indices.append(i)
        except Exception as e:
            continue

    print(f"\n✅ Feature extraction complete: {len(features_list):,} valid URLs")

    # Create feature matrix
    X_test = np.array(features_list)
    y_test = y_test[valid_indices]

    print(f"Feature matrix shape: {X_test.shape}")
    print(f"Labels shape: {y_test.shape}")

    # Scale features
    X_test_scaled = detector.ensemble_classifier.scaler.transform(X_test)

    # Step 5: Test individual models
    print_header("STEP 5: Testing Individual Models")

    results = []
    individual_probas = {}

    # Test XGBoost
    metrics_xgb, proba_xgb = test_individual_model(
        detector.ensemble_classifier.xgb_model,
        X_test_scaled,
        y_test,
        "XGBoost"
    )
    results.append(metrics_xgb)
    individual_probas['xgboost'] = proba_xgb

    # Test Random Forest
    metrics_rf, proba_rf = test_individual_model(
        detector.ensemble_classifier.rf_model,
        X_test_scaled,
        y_test,
        "Random Forest"
    )
    results.append(metrics_rf)
    individual_probas['random_forest'] = proba_rf

    # Test Neural Network
    metrics_nn, proba_nn = test_individual_model(
        detector.ensemble_classifier.nn_model,
        X_test_scaled,
        y_test,
        "Neural Network"
    )
    results.append(metrics_nn)
    individual_probas['neural_network'] = proba_nn

    # Step 6: Test weighted ensemble
    print_header("STEP 6: Testing Weighted Ensemble")

    metrics_ensemble = test_ensemble(detector, X_test_scaled, y_test, individual_probas)
    results.append(metrics_ensemble)

    # Step 7: Compare all models
    df_comparison = compare_models(results)

    # Step 8: Save results
    print_header("STEP 8: Saving Results")

    output_file = 'performance_comparison_results.csv'
    df_comparison.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")

    print_header("TEST COMPLETE")
    print("Summary:")
    print(f"  ✓ Tested {len(test_urls):,} URLs from test set")
    print(f"  ✓ Evaluated 4 models (3 individual + 1 ensemble)")
    print(f"  ✓ Best model: {df_comparison.iloc[0]['model']}")
    print(f"  ✓ Best accuracy: {df_comparison.iloc[0]['accuracy']:.4f}")
    print(f"\nResults file: {output_file}")


if __name__ == "__main__":
    main()
