"""Evaluate model using ONLY lexical features (39 features)"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import argparse
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.phishing_detector.features.extractor import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Define host-based features to exclude
HOST_FEATURES = {
    # DNS features
    'has_dns_a_record', 'num_dns_a_records',
    'has_dns_mx_record', 'num_dns_mx_records',
    'has_dns_ns_record', 'num_dns_ns_records',
    'has_dns_txt_record', 'has_ptr_record',
    # SSL features
    'has_ssl_cert', 'ssl_cert_valid', 'ssl_days_to_expire',
    'ssl_cert_expires_soon', 'ssl_cert_age_days', 'ssl_cert_is_new',
    'ssl_num_san',
    # Port features (these are actually in lexical, but we'll keep them)
    # 'has_port', 'uses_standard_port', 'uses_non_standard_port',
}

def evaluate_lexical_only(file_path: str, test_size: float = 0.2):
    """Evaluate model using only lexical features"""
    
    print("=" * 70)
    print("LEXICAL-ONLY FEATURE EVALUATION (39 features)")
    print("=" * 70)
    
    # 1. Load Data
    logger.info(f"\n📂 Loading test data from {file_path}...")
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        if "url" not in df.columns or "label" not in df.columns:
            raise ValueError(f"File must contain 'url' and 'label' columns. Found: {df.columns.tolist()}")
            
        df = df.dropna(subset=['url', 'label'])
        urls = df["url"].tolist()
        labels = df["label"].astype(int).tolist()
        logger.info(f"✓ Loaded {len(urls)} URLs")
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        sys.exit(1)

    # 2. Extract Features (lexical only - DNS/SSL disabled)
    logger.info("\n🔍 Extracting LEXICAL features only (DNS/SSL disabled)...")
    feature_extractor = FeatureExtractor(
        enable_dns_lookup=False,  # Disable for lexical-only
        enable_ssl_lookup=False,
        enable_whois_lookup=False,
        timeout=5,
    )
    
    features_list = []
    valid_labels = []
    
    for i, (url, label) in enumerate(zip(urls, labels)):
        try:
            if i % 100 == 0:
                print(f"   Processing {i}/{len(urls)}...", end="\r")
            features = feature_extractor.extract_features(url)
            features_list.append(features)
            valid_labels.append(int(label))
        except Exception as e:
            logger.warning(f"Skipping {url}: {e}")
            
    print(f"   Processing {len(urls)}/{len(urls)}... Done!")

    # 3. Filter to lexical features only
    df_features = pd.DataFrame(features_list)
    
    # Remove host-based features
    lexical_features = [col for col in df_features.columns if col not in HOST_FEATURES]
    df_lexical = df_features[lexical_features]
    
    logger.info(f"✓ Extracted {len(lexical_features)} lexical features")
    logger.info(f"   Excluded {len(HOST_FEATURES)} host-based features")
    
    X = df_lexical.values
    y = np.array(valid_labels, dtype=np.int32)

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"\n📊 Dataset split:")
    logger.info(f"   Training: {len(X_train)} samples")
    logger.info(f"   Testing:  {len(X_test)} samples")

    # 5. Train model on lexical features only
    logger.info("\n🎯 Training XGBoost on LEXICAL features only...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, verbose=False)
    logger.info("✓ Training complete")

    # 6. Evaluate
    logger.info("\n📈 Evaluating model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Print Results
    print("\n" + "=" * 70)
    print("LEXICAL-ONLY EVALUATION RESULTS")
    print("=" * 70)
    print(f"Features Used:       39 LEXICAL features only")
    print(f"Accuracy:            {accuracy:.4%}")
    print(f"Precision:           {precision:.4%}")
    print(f"Recall (Sensitivity):{recall:.4%}")
    print(f"F1-Score:            {f1:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print("-" * 70)
    print(f"False Positive Rate: {fpr:.4%} (LOWER IS BETTER)")
    print("-" * 70)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Safe correctly identified):      {tn}")
    print(f"False Positives (Safe wrongly blocked):          {fp}  <-- CRITICAL")
    print(f"False Negatives (Phishing missed):               {fn}")
    print(f"True Positives (Phishing correctly blocked):     {tp}")
    
    # Feature importance
    print("\n" + "=" * 70)
    print("TOP 10 MOST IMPORTANT LEXICAL FEATURES")
    print("=" * 70)
    feature_importance = sorted(
        zip(lexical_features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (feature, importance) in enumerate(feature_importance[:10], 1):
        print(f"{i:2d}. {feature:30s} {importance:.4f}")
    
    print("=" * 70)
    
    # Save misclassified
    misclassified = []
    for i, (true_label, pred_label, proba) in enumerate(zip(y_test, y_pred, y_pred_proba)):
        if true_label != pred_label:
            # Get original URL index
            test_indices = list(range(len(X)))[len(X_train):]
            original_idx = test_indices[i]
            
            misclassified.append({
                "url": urls[original_idx],
                "true_label": "Phishing" if true_label == 1 else "Legitimate",
                "predicted_label": "Phishing" if pred_label == 1 else "Legitimate",
                "confidence": proba,
                "error_type": "False Positive" if pred_label == 1 else "False Negative"
            })

    if misclassified:
        df_errors = pd.DataFrame(misclassified)
        error_file = "misclassified_lexical_only.csv"
        df_errors.to_csv(error_file, index=False)
        
        print(f"\n⚠️  Found {len(misclassified)} misclassified URLs.")
        print(f"   Details saved to: {error_file}")
    else:
        print("\n✅ PERFECT SCORE! No misclassified URLs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model using only lexical features")
    parser.add_argument("data", type=str, help="Path to test data CSV file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")
    
    args = parser.parse_args()
    evaluate_lexical_only(args.data, args.test_size)
