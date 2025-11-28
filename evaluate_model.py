"""Evaluate model against a specific test file"""

import warnings
# Suppress urllib3 SSL warning BEFORE other imports
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
from src.phishing_detector.detector import PhishingDetector
from src.phishing_detector.features.extractor import FeatureExtractor
from config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

def evaluate_custom_file(file_path: str):
    """Evaluate model against a custom file"""
    
    # 1. Load Data
    logger.info(f"Loading test data from {file_path}...")
    try:
        # Read CSV with error handling
        df = pd.read_csv(file_path, on_bad_lines='skip')
        
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Ensure columns exist
        if "url" not in df.columns or "label" not in df.columns:
            raise ValueError(f"File must contain 'url' and 'label' columns. Found: {df.columns.tolist()}")
            
        # Drop rows with missing values
        df = df.dropna(subset=['url', 'label'])
        
        urls = df["url"].tolist()
        labels = df["label"].tolist()
        logger.info(f"Loaded {len(urls)} URLs")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # 2. Initialize Detector
    detector = PhishingDetector(
        enable_dns_lookup=False, # Disable for speed
    )
    
    # 3. Load Model
    try:
        detector.load(str(settings.models_dir))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Please train the model first using 'python training/train.py'")
        sys.exit(1)

    # 4. Extract Features
    logger.info("Extracting features (this may take a moment)...")
    feature_extractor = FeatureExtractor(
        enable_dns_lookup=False,
        enable_ssl_lookup=False, # Disable for speed
        enable_whois_lookup=False,
        timeout=5,
    )
    
    features_list = []
    valid_labels = []
    
    for i, (url, label) in enumerate(zip(urls, labels)):
        try:
            if i % 10 == 0:
                print(f"Processing {i}/{len(urls)}...", end="\r")
            features = feature_extractor.extract_features(url)
            features_list.append(features)
            valid_labels.append(int(label))
        except Exception as e:
            logger.warning(f"Skipping {url}: {e}")
            
    print(f"Processing {len(urls)}/{len(urls)}... Done!")

    # Convert to numpy
    df_features = pd.DataFrame(features_list)
    X_test = df_features.values
    y_test = np.array(valid_labels, dtype=np.int32)

    # 5. Evaluate and Error Analysis
    logger.info("Evaluating model...")
    
    # Get predictions
    y_pred_proba = detector.model.predict_proba(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # False Positive Rate (Fall-out)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Print Results
    print("\n" + "=" * 60)
    print("DETAILED EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall (Sensitivity):{recall:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print("-" * 30)
    print(f"False Positive Rate: {fpr:.4f} (LOWER IS BETTER)")
    print("-" * 30)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Safe correctly identified):      {tn}")
    print(f"False Positives (Safe wrongly blocked):          {fp}  <-- CRITICAL")
    print(f"False Negatives (Phishing missed):               {fn}")
    print(f"True Positives (Phishing correctly blocked):     {tp}")
    print("=" * 60)

    # Identify Misclassified URLs
    misclassified = []
    for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
        if true_label != pred_label:
            misclassified.append({
                "url": urls[i],
                "true_label": "Phishing" if true_label == 1 else "Legitimate",
                "predicted_label": "Phishing" if pred_label == 1 else "Legitimate",
                "confidence": y_pred_proba[i],
                "error_type": "False Positive" if pred_label == 1 else "False Negative"
            })

    # Save Misclassified
    if misclassified:
        df_errors = pd.DataFrame(misclassified)
        error_file = "misclassified_urls.csv"
        df_errors.to_csv(error_file, index=False)
        
        print(f"\n⚠️  Found {len(misclassified)} misclassified URLs.")
        print(f"   Details saved to: {error_file}")
        
        print("\nTop 5 False Positives (Safe sites blocked):")
        fps = df_errors[df_errors["error_type"] == "False Positive"].head(5)
        if not fps.empty:
            for _, row in fps.iterrows():
                print(f"   - {row['url']} (Score: {row['confidence']:.2f})")
        else:
            print("   None")

        print("\nTop 5 False Negatives (Phishing sites missed):")
        fns = df_errors[df_errors["error_type"] == "False Negative"].head(5)
        if not fns.empty:
            for _, row in fns.iterrows():
                print(f"   - {row['url']} (Conf: {row['confidence']:.2f})")
        else:
            print("   None")
    else:
        print("\n✅  PERFECT SCORE! No misclassified URLs.")
    
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_csv.py <path_to_csv_or_txt>")
        sys.exit(1)
        
    evaluate_custom_file(sys.argv[1])
