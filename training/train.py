"""Improved training script with better sample data"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_improved_sample_data() -> pd.DataFrame:
    """Load improved sample training data with more diverse URLs"""
    logger.info("Generating improved sample training data...")

    # Create DataFrame
    df = pd.DataFrame(
        {
            "url": legitimate_urls + phishing_urls,
            "label": [0] * len(legitimate_urls) + [1] * len(phishing_urls),
        }
    )

    logger.info(f"Generated {len(legitimate_urls)} legitimate URLs")
    logger.info(f"Generated {len(phishing_urls)} phishing URLs")
    logger.info(f"Total: {len(df)} URLs")

    return df


def extract_features_from_urls(urls: list, labels: list) -> tuple:
    """Extract features from URLs"""
    logger.info("Extracting features from URLs...")

    feature_extractor = FeatureExtractor(
        enable_dns_lookup=False,  # Disable for faster training
        enable_ssl_lookup=False,  # Disable for faster training
        enable_whois_lookup=False,
        timeout=5,
    )

    # Extract features
    features_list = []
    valid_labels = []

    for url, label in zip(urls, labels):
        try:
            features = feature_extractor.extract_features(url)
            features_list.append(features)
            # Ensure label is integer
            valid_labels.append(int(label))
        except Exception as e:
            logger.warning(f"Failed to extract features from {url}: {e}")

    # Convert to DataFrame
    df_features = pd.DataFrame(features_list)

    # Get feature names
    feature_names = df_features.columns.tolist()

    # Convert to numpy arrays
    X = df_features.values
    y = np.array(valid_labels, dtype=np.int32)

    logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} URLs")

    return X, y, feature_names


def evaluate_model(detector: PhishingDetector, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance"""
    logger.info("Evaluating model...")

    # Get predictions
    y_pred_proba = detector.model.predict_proba(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    logger.info("\n" + "=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info("\nConfusion Matrix:")
    logger.info(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    logger.info(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
    logger.info("=" * 50)

    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    return metrics


def train_model(
    data_path: str = None,
    output_dir: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: int = None,
) -> None:
    """Main training function"""

    # Use settings defaults if not provided
    output_dir = output_dir or settings.model_save_path
    random_state = random_state or settings.random_seed

    # Load data
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, on_bad_lines='skip')

        # Limit rows if specified
        if max_rows:
            logger.info(f"Limiting to first {max_rows} rows")
            df = df.head(max_rows)

        # Convert labels to integers
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        # Filter valid labels (0 or 1)
        df = df[df['label'].isin([0, 1])]

        logger.info(f"Loaded {len(df)} total URLs")
        logger.info(f"  Legitimate (0): {len(df[df['label'] == 0])}")
        logger.info(f"  Phishing (1): {len(df[df['label'] == 1])}")

        urls = df["url"].tolist()
        labels = df["label"].tolist()
    else:
        logger.info("Using improved sample data")
        df = load_improved_sample_data()
        urls = df["url"].tolist()
        labels = df["label"].tolist()

    # Extract features
    X, y, feature_names = extract_features_from_urls(urls, labels)

    # Use all data for training (no split)
    X_train, y_train = X, y
    
    logger.info(f"Training set: {X_train.shape[0]} samples (Using 100% of data)")
    logger.info(f"Phishing ratio: {y_train.mean():.2%}")

    # Initialize detector
    detector = PhishingDetector(
        enable_dns_lookup=True,  # Enable DNS/SSL lookups to use all 57 features
        random_state=random_state,
    )

    # Train model
    logger.info("Starting training on full dataset...")
    detector.train(X_train, y_train, None, None, feature_names)

    # Skip evaluation during training
    logger.info("\n" + "=" * 50)
    logger.info("NOTE: Model evaluation skipped during training.")
    logger.info("To evaluate the model, please run:")
    logger.info("python evaluate_model.py <path_to_test_data.csv>")
    logger.info("=" * 50)

    # Get feature importance
    logger.info("\nTop 10 Most Important Features:")
    feature_importance = detector.get_feature_importance()
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        logger.info(f"{i}. {feature}: {importance:.4f}")

    # Save model
    logger.info(f"\nSaving model to {output_dir}")
    detector.save(output_dir)

    logger.info("\nTraining completed successfully!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train phishing detector model with improved data")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data CSV file (url, label columns)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to use from CSV (default: all)",
    )

    args = parser.parse_args()

    try:
        train_model(
            data_path=args.data,
            output_dir=args.output,
            test_size=args.test_size,
            random_state=args.random_state,
            max_rows=args.max_rows,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
