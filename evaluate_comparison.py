"""Compare model performance across different feature sets"""

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

# Define host-based features
HOST_FEATURES = {
    'has_dns_a_record', 'num_dns_a_records',
    'has_dns_mx_record', 'num_dns_mx_records',
    'has_dns_ns_record', 'num_dns_ns_records',
    'has_dns_txt_record', 'has_ptr_record',
    'has_ssl_cert', 'ssl_cert_valid', 'ssl_days_to_expire',
    'ssl_cert_expires_soon', 'ssl_cert_age_days', 'ssl_cert_is_new',
    'ssl_num_san',
    'has_port', 'uses_standard_port', 'uses_non_standard_port',
}

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names, model_name):
    """Train and evaluate a model"""
    
    # Train
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
    
    # Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Feature importance
    feature_importance = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        'model_name': model_name,
        'num_features': len(feature_names),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'feature_importance': feature_importance,
    }

def compare_models(file_path: str, test_size: float = 0.2, enable_dns: bool = False):
    """Compare all three feature configurations"""
    
    print("=" * 80)
    print("FEATURE SET COMPARISON: LEXICAL vs HOST-BASED vs COMBINED")
    print("=" * 80)
    
    # 1. Load Data
    logger.info(f"\n📂 Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        if "url" not in df.columns or "label" not in df.columns:
            raise ValueError(f"File must contain 'url' and 'label' columns")
            
        df = df.dropna(subset=['url', 'label'])
        urls = df["url"].tolist()
        labels = df["label"].astype(int).tolist()
        logger.info(f"✓ Loaded {len(urls)} URLs")
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        sys.exit(1)

    # 2. Extract Features
    if enable_dns:
        logger.info("\n🔍 Extracting ALL features (DNS/SSL enabled)...")
        logger.info("   ⏳ This may take several minutes...")
    else:
        logger.info("\n🔍 Extracting features (DNS/SSL disabled for speed)...")
        logger.info("   ⚠️  Host-based features will have default values")
    
    feature_extractor = FeatureExtractor(
        enable_dns_lookup=enable_dns,
        enable_ssl_lookup=enable_dns,
        enable_whois_lookup=False,
        timeout=5,
    )
    
    features_list = []
    valid_labels = []
    
    for i, (url, label) in enumerate(zip(urls, labels)):
        try:
            if i % (10 if enable_dns else 100) == 0:
                print(f"   Processing {i}/{len(urls)}...", end="\r")
            features = feature_extractor.extract_features(url)
            features_list.append(features)
            valid_labels.append(int(label))
        except Exception as e:
            logger.warning(f"Skipping {url}: {e}")
            
    print(f"   Processing {len(urls)}/{len(urls)}... Done!          ")
    
    df_features = pd.DataFrame(features_list)
    y = np.array(valid_labels, dtype=np.int32)
    
    logger.info(f"✓ Extracted {len(df_features.columns)} total features")

    # 3. Prepare feature sets
    all_features = df_features.columns.tolist()
    lexical_features = [col for col in all_features if col not in HOST_FEATURES]
    host_features = [col for col in all_features if col in HOST_FEATURES]
    
    logger.info(f"   - Lexical features: {len(lexical_features)}")
    logger.info(f"   - Host features: {len(host_features)}")
    logger.info(f"   - Combined features: {len(all_features)}")
    
    # 4. Train and evaluate three models
    results = []
    
    print("\n" + "=" * 80)
    print("TRAINING MODELS...")
    print("=" * 80)
    
    # Model 1: Lexical only
    logger.info("\n🎯 Training Model 1: LEXICAL-ONLY (39 features)...")
    X_lexical = df_features[lexical_features].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_lexical, y, test_size=test_size, random_state=42, stratify=y
    )
    result_lexical = train_and_evaluate(
        X_train, X_test, y_train, y_test, lexical_features, "Lexical-Only"
    )
    results.append(result_lexical)
    logger.info("✓ Complete")
    
    # Model 2: Host-based only
    logger.info("\n🎯 Training Model 2: HOST-BASED-ONLY (18 features)...")
    X_host = df_features[host_features].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_host, y, test_size=test_size, random_state=42, stratify=y
    )
    result_host = train_and_evaluate(
        X_train, X_test, y_train, y_test, host_features, "Host-Based-Only"
    )
    results.append(result_host)
    logger.info("✓ Complete")
    
    # Model 3: Combined
    logger.info("\n🎯 Training Model 3: COMBINED (57 features)...")
    X_combined = df_features[all_features].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=test_size, random_state=42, stratify=y
    )
    result_combined = train_and_evaluate(
        X_train, X_test, y_train, y_test, all_features, "Combined"
    )
    results.append(result_combined)
    logger.info("✓ Complete")
    
    # 5. Display comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Metrics table
    print(f"\n{'Model':<20} {'Features':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'FPR':>10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model_name']:<20} "
              f"{result['num_features']:>10} "
              f"{result['accuracy']:>10.4f} "
              f"{result['precision']:>10.4f} "
              f"{result['recall']:>10.4f} "
              f"{result['f1']:>10.4f} "
              f"{result['roc_auc']:>10.4f} "
              f"{result['fpr']:>10.4f}")
    
    # Confusion matrices
    print("\n" + "=" * 80)
    print("CONFUSION MATRICES")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['model_name']}:")
        print(f"  TN: {result['tn']:>5}  FP: {result['fp']:>5}")
        print(f"  FN: {result['fn']:>5}  TP: {result['tp']:>5}")
    
    # Top features for each model
    print("\n" + "=" * 80)
    print("TOP 5 FEATURES BY MODEL")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['model_name']}:")
        for i, (feature, importance) in enumerate(result['feature_importance'][:5], 1):
            print(f"  {i}. {feature:<35} {importance:.4f}")
    
    # Best model
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    best_f1 = max(results, key=lambda x: x['f1'])
    best_fpr = min(results, key=lambda x: x['fpr'])
    
    print(f"\n🏆 Best Accuracy:  {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
    print(f"🏆 Best F1-Score:  {best_f1['model_name']} ({best_f1['f1']:.4f})")
    print(f"🏆 Lowest FPR:     {best_fpr['model_name']} ({best_fpr['fpr']:.4f})")
    
    if not enable_dns:
        print("\n⚠️  NOTE: DNS/SSL lookups were disabled for speed.")
        print("   Host-based features have default values and may not be accurate.")
        print("   Run with --enable-dns for accurate host-based evaluation.")
    
    print("=" * 80)
    
    # Save comparison to CSV
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Features': r['num_features'],
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1-Score': r['f1'],
            'ROC-AUC': r['roc_auc'],
            'FPR': r['fpr'],
            'TN': r['tn'],
            'FP': r['fp'],
            'FN': r['fn'],
            'TP': r['tp'],
        }
        for r in results
    ])
    
    output_file = "feature_comparison_results.csv"
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"\n💾 Comparison saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model performance across feature sets")
    parser.add_argument("data", type=str, help="Path to test data CSV file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument("--enable-dns", action="store_true", help="Enable DNS/SSL lookups (slow but accurate)")
    
    args = parser.parse_args()
    compare_models(args.data, args.test_size, args.enable_dns)
