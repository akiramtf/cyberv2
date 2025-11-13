"""Training script for phishing detector"""

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


def load_sample_data() -> pd.DataFrame:
    """Load or generate sample training data"""
    logger.info("Generating sample training data...")

    # Sample legitimate URLs
    legitimate_urls = [
        "https://www.google.com",
        "https://www.wikipedia.org",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.python.org",
        "https://www.tensorflow.org",
        "https://www.scikit-learn.org",
        "https://www.numpy.org",
        "https://www.pandas.pydata.org",
        "https://www.docker.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.amazon.com",
        "https://www.netflix.com",
        "https://www.youtube.com",
        "https://www.linkedin.com",
        "https://www.twitter.com",
        "https://www.facebook.com",
        "https://www.reddit.com",
        "https://www.medium.com",
    ]

    # Sample phishing URLs (these are safe examples, not real phishing)
    phishing_urls = [
        "http://paypal-verify.suspicious-domain.tk/login.php",
        "http://192.168.1.1/bank/login",
        "http://secure-amazon-update.xyz/account",
        "http://apple-id-locked.ml/verify",
        "http://microsoft-security-alert.gq/signin",
        "http://bit.ly/3xYz@suspicious",
        "http://banking-secure-login.click/auth",
        "http://paypal.com-verify.work/update",
        "http://amazon.security-check.top/account",
        "http://ebay-suspended.racing/verify",
        "http://google-security.download/signin",
        "http://facebook-verify.link/confirm",
        "http://netflix-payment.bid/update",
        "http://linkedin-security.date/verify",
        "http://twitter-suspended.accountant/appeal",
        "http://instagram-verify.science/confirm",
        "http://dropbox-storage-full.win/upgrade",
        "http://adobe-update-required.click/download",
        "http://spotify-premium-free.top/claim",
        "http://apple-icloud-storage.xyz/upgrade",
    ]

    # Create DataFrame
    df = pd.DataFrame(
        {
            "url": legitimate_urls + phishing_urls,
            "label": [0] * len(legitimate_urls) + [1] * len(phishing_urls),
        }
    )

    return df


def extract_features_from_urls(urls: list, labels: list) -> tuple:
    """Extract features from URLs"""
    logger.info("Extracting features from URLs...")

    feature_extractor = FeatureExtractor(
        enable_dns_lookup=settings.enable_dns_lookup,
        enable_whois_lookup=settings.enable_whois_lookup,
        timeout=settings.request_timeout,
    )

    # Extract features
    features_list = []
    valid_labels = []

    for url, label in zip(urls, labels):
        try:
            features = feature_extractor.extract_features(url)
            features_list.append(features)
            valid_labels.append(label)
        except Exception as e:
            logger.warning(f"Failed to extract features from {url}: {e}")

    # Convert to DataFrame
    df_features = pd.DataFrame(features_list)

    # Get feature names
    feature_names = df_features.columns.tolist()

    # Convert to numpy arrays
    X = df_features.values
    y = np.array(valid_labels)

    logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} URLs")

    return X, y, feature_names


def evaluate_model(detector: PhishingDetector, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance"""
    logger.info("Evaluating model...")

    # Get predictions
    y_pred_proba = detector.ensemble_classifier.predict_proba(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
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
) -> None:
    """Main training function"""

    # Use settings defaults if not provided
    output_dir = output_dir or settings.model_save_path
    random_state = random_state or settings.random_seed

    # Load data
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        urls = df["url"].tolist()
        labels = df["label"].tolist()
    else:
        logger.info("Using sample data for demonstration")
        df = load_sample_data()
        urls = df["url"].tolist()
        labels = df["label"].tolist()

    # Extract features
    X, y, feature_names = extract_features_from_urls(urls, labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Phishing ratio in training: {y_train.mean():.2%}")

    # Initialize detector
    detector = PhishingDetector(
        enable_zero_day=settings.enable_zero_day_detection,
        enable_dns_lookup=settings.enable_dns_lookup,
        anomaly_threshold=settings.anomaly_threshold,
        random_state=random_state,
    )

    # Train model
    logger.info("Starting training...")
    history = detector.train(X_train, y_train, X_test, y_test, feature_names)

    # Evaluate model
    metrics = evaluate_model(detector, X_test, y_test)

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
    parser = argparse.ArgumentParser(description="Train phishing detector model")
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

    args = parser.parse_args()

    try:
        train_model(
            data_path=args.data,
            output_dir=args.output,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
