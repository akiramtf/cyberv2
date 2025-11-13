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

    # More legitimate URLs (popular, trusted sites)
    legitimate_urls = [
        # Search engines and portals
        "https://www.google.com",
        "https://www.bing.com",
        "https://www.yahoo.com",
        "https://duckduckgo.com",
        # Social media
        "https://www.facebook.com",
        "https://www.twitter.com",
        "https://www.linkedin.com",
        "https://www.instagram.com",
        "https://www.reddit.com",
        # Tech companies
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.amazon.com",
        "https://www.netflix.com",
        "https://www.adobe.com",
        # Development
        "https://www.github.com",
        "https://stackoverflow.com",
        "https://www.python.org",
        "https://www.tensorflow.org",
        "https://www.docker.com",
        # News and media
        "https://www.cnn.com",
        "https://www.bbc.com",
        "https://www.nytimes.com",
        "https://www.theguardian.com",
        # Education
        "https://www.wikipedia.org",
        "https://www.coursera.org",
        "https://www.edx.org",
        "https://www.khanacademy.org",
        # E-commerce
        "https://www.ebay.com",
        "https://www.etsy.com",
        "https://www.walmart.com",
        "https://www.target.com",
        # Finance
        "https://www.paypal.com",
        "https://www.stripe.com",
        "https://www.chase.com",
        "https://www.bankofamerica.com",
        # Cloud services
        "https://www.dropbox.com",
        "https://drive.google.com",
        "https://onedrive.live.com",
        "https://www.icloud.com",
        # Others
        "https://www.spotify.com",
        "https://www.zoom.us",
        "https://www.slack.com",
        "https://www.trello.com",
        "https://www.medium.com",
        "https://www.wordpress.com",
        "https://www.blogger.com",
        "https://www.tumblr.com",
        "https://www.pinterest.com",
        "https://www.twitch.tv",
        "https://www.youtube.com",
    ]

    # Phishing-like URLs (fake/suspicious patterns)
    phishing_urls = [
        # IP addresses
        "http://192.168.1.1/bank/login",
        "http://10.0.0.1/paypal/signin",
        "http://172.16.0.1/secure/account",
        # Suspicious TLDs
        "http://paypal-verify.suspicious-domain.tk/login.php",
        "http://apple-id-locked.ml/verify",
        "http://microsoft-security-alert.gq/signin",
        "http://banking-secure-login.click/auth",
        "http://amazon.security-check.top/account",
        "http://ebay-suspended.racing/verify",
        "http://google-security.download/signin",
        "http://facebook-verify.link/confirm",
        "http://netflix-payment.bid/update",
        "http://linkedin-security.date/verify",
        "http://twitter-suspended.accountant/appeal",
        "http://instagram-verify.science/confirm",
        "http://dropbox-storage-full.win/upgrade",
        "http://adobe-update-required.xyz/download",
        "http://spotify-premium-free.top/claim",
        "http://apple-icloud-storage.work/upgrade",
        # Misspellings and typos
        "http://gooogle.com/login",
        "http://faceboook.com/signin",
        "http://amaz0n.com/account",
        "http://paypa1.com/verify",
        "http://app1e.com/icloud",
        # Suspicious keywords
        "http://secure-paypal-update.com/login",
        "http://account-verify-amazon.net/confirm",
        "http://microsoft-security-team.org/alert",
        "http://apple-account-locked.info/unlock",
        "http://netflix-billing-update.biz/payment",
        "http://bank-account-suspended.online/reactivate",
        # URL shorteners with suspicious patterns
        "http://bit.ly/3xYz@suspicious",
        "http://tinyurl.com/verify-account-now",
        # Subdomain spoofing
        "http://paypal.com-verify.work/update",
        "http://apple.com-support.xyz/help",
        "http://amazon.com-secure.top/account",
        "http://microsoft.com-login.info/signin",
        # Long suspicious URLs
        "http://verify-your-paypal-account-now-or-suspended.tk/login.php",
        "http://amazon-account-verification-required-urgent.ml/verify",
        "http://apple-id-security-alert-action-needed.gq/confirm",
        # Double slashes and special chars
        "http://paypal.com//login",
        "http://amazon.com@phishing.tk",
        "http://secure-login.com//..//bank",
        # Phishing with ports
        "http://paypal.com:8080/login",
        "http://amazon.com:3000/account",
        # Mixed legitimate and suspicious
        "http://secure-update-microsoft.tk/signin",
        "http://verify-google-account.ml/confirm",
        "http://apple-support-team.gq/help",
        "http://netflix-reactivate.xyz/payment",
    ]

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
        logger.info("Using improved sample data")
        df = load_improved_sample_data()
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

    # Initialize detector with adjusted settings
    detector = PhishingDetector(
        enable_zero_day=True,
        enable_dns_lookup=False,
        anomaly_threshold=0.5,  # Increased threshold to reduce false positives
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
