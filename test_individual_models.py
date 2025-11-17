"""Test individual model performance vs ensemble"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.phishing_detector.detector import PhishingDetector
from src.phishing_detector.features.extractor import FeatureExtractor

def main():
    print("Loading dataset...")
    df = pd.read_csv('dataset2.csv', on_bad_lines='skip').head(10000)  # Use 10k for speed

    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor(enable_dns_lookup=False)

    features_list = []
    labels = []

    for _, row in df.iterrows():
        try:
            features = feature_extractor.extract_features(row['url'])
            features_list.append(features)
            labels.append(int(row['label']))
        except:
            pass

    X = pd.DataFrame(features_list).values
    y = np.array(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train detector
    print("\nTraining ensemble detector...")
    detector = PhishingDetector(enable_dns_lookup=False)
    detector.train(X_train, y_train, X_test, y_test)

    # Test individual models
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL PERFORMANCE COMPARISON")
    print("="*60)

    # XGBoost only
    y_pred_xgb = detector.ensemble_classifier.xgboost_model.predict(X_test)
    y_proba_xgb = detector.ensemble_classifier.xgboost_model.predict_proba(X_test)[:, 1]

    print("\n1. XGBoost (40% weight):")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred_xgb):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_xgb):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_xgb):.4f}")

    # Random Forest only
    y_pred_rf = detector.ensemble_classifier.random_forest_model.predict(X_test)
    y_proba_rf = detector.ensemble_classifier.random_forest_model.predict_proba(X_test)[:, 1]

    print("\n2. Random Forest (30% weight):")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_rf):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_rf):.4f}")

    # Neural Network only
    y_proba_nn = detector.ensemble_classifier.neural_network_model.predict(X_test, verbose=0).flatten()
    y_pred_nn = (y_proba_nn >= 0.5).astype(int)

    print("\n3. Neural Network (30% weight):")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_nn):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred_nn):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_nn):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_nn):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_nn):.4f}")

    # Weighted Ensemble
    y_proba_ensemble = detector.ensemble_classifier.predict_proba(X_test)
    y_pred_ensemble = (y_proba_ensemble >= 0.5).astype(int)

    print("\n4. Weighted Ensemble (40-30-30):")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_ensemble):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred_ensemble):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_ensemble):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_ensemble):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_ensemble):.4f}")

    print("\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)

    acc_improvement = accuracy_score(y_test, y_pred_ensemble) - max(
        accuracy_score(y_test, y_pred_xgb),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_nn)
    )

    print(f"\nEnsemble improvement over best individual model:")
    print(f"  Accuracy gain: +{acc_improvement:.4f} ({acc_improvement*100:.2f}%)")

if __name__ == "__main__":
    main()
