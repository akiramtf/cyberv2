"""Test individual model performance using held-out test set (no data leakage)"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.phishing_detector.detector import PhishingDetector
from src.phishing_detector.features.extractor import FeatureExtractor

def main():
    # Check if trained models exist
    model_path = 'models/trained'
    if not os.path.exists(os.path.join(model_path, 'xgboost_model.json')):
        print(f"Error: No trained models found in {model_path}/")
        print("Please train models first using:")
        print("  python training/train.py --data dataset2.csv --output models/trained")
        return

    print("="*70)
    print("PROPER MODEL EVALUATION (Avoiding Data Leakage)")
    print("="*70)
    print("\nStrategy: Split dataset FIRST, then train/test on separate sets")
    print()

    # Load full dataset
    print("Loading full dataset...")
    df = pd.read_csv('dataset2.csv', on_bad_lines='skip')
    print(f"Total URLs: {len(df)}")

    # Split FIRST (80/20 split with same random state as training)
    # This ensures train/test separation
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    print(f"Training set: {len(train_df)} URLs")
    print(f"Test set: {len(test_df)} URLs (HELD-OUT, unseen during training)")

    # Use only a subset of test set for faster evaluation
    test_sample = test_df.sample(n=min(2000, len(test_df)), random_state=42)
    print(f"Using {len(test_sample)} URLs from test set for evaluation\n")

    # Extract features from TEST SET ONLY
    print("Extracting features from TEST set...")
    feature_extractor = FeatureExtractor(enable_dns_lookup=False)

    features_list = []
    labels = []

    for idx, row in test_sample.iterrows():
        try:
            features = feature_extractor.extract_features(row['url'])
            features_list.append(features)
            labels.append(int(row['label']))

            if (len(features_list) + 1) % 500 == 0:
                print(f"  Processed {len(features_list)} URLs...")
        except:
            pass

    print(f"  Completed: {len(features_list)} URLs extracted\n")

    X_test = pd.DataFrame(features_list).values
    y_test = np.array(labels)

    # Load pre-trained models
    print(f"Loading pre-trained models from {model_path}...")
    detector = PhishingDetector(enable_dns_lookup=False)
    detector.load(model_path)

    # Test individual models
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL PERFORMANCE COMPARISON")
    print("="*70)

    # XGBoost only
    y_pred_xgb = detector.ensemble_classifier.xgb_model.predict(X_test)
    y_proba_xgb = detector.ensemble_classifier.xgb_model.predict_proba(X_test)[:, 1]

    print("\n1. XGBoost (40% weight):")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred_xgb, zero_division=0):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_xgb, zero_division=0):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_xgb, zero_division=0):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_xgb):.4f}")

    # Random Forest only
    y_pred_rf = detector.ensemble_classifier.rf_model.predict(X_test)
    y_proba_rf = detector.ensemble_classifier.rf_model.predict_proba(X_test)[:, 1]

    print("\n2. Random Forest (30% weight):")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_rf):.4f}")

    # Neural Network only
    y_proba_nn = detector.ensemble_classifier.nn_model.predict(X_test, verbose=0).flatten()
    y_pred_nn = (y_proba_nn >= 0.5).astype(int)

    print("\n3. Neural Network (30% weight):")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_nn):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred_nn, zero_division=0):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_nn, zero_division=0):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_nn, zero_division=0):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_nn):.4f}")

    # Weighted Ensemble
    y_proba_ensemble = detector.ensemble_classifier.predict_proba(X_test)
    y_pred_ensemble = (y_proba_ensemble >= 0.5).astype(int)

    print("\n4. Weighted Ensemble (40-30-30):")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_ensemble):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred_ensemble, zero_division=0):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_ensemble, zero_division=0):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_ensemble, zero_division=0):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_ensemble):.4f}")

    print("\n" + "="*70)
    print("IMPROVEMENT ANALYSIS")
    print("="*70)

    acc_improvement = accuracy_score(y_test, y_pred_ensemble) - max(
        accuracy_score(y_test, y_pred_xgb),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_nn)
    )

    print(f"\nEnsemble improvement over best individual model:")
    print(f"  Accuracy gain: +{acc_improvement:.4f} ({acc_improvement*100:.2f}%)")

    print("\n" + "="*70)
    print(f"✓ Using HELD-OUT test set (no data leakage)")
    print(f"✓ Models trained on: {len(train_df)} URLs")
    print(f"✓ Tested on: {len(features_list)} UNSEEN URLs")
    print("="*70)

if __name__ == "__main__":
    main()
