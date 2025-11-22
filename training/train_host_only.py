"""Train model using ONLY host-based features (18 features)"""

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

from src.phishing_detector.features.extractor import FeatureExtractor
from src.phishing_detector.models.phishing_model import PhishingModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Host-based features to keep
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

def train_host_only(
    data_path: str,
    output_dir: str = "models/host_only",
    max_rows: int = None,
):
    """Train model using only host-based features"""
    
    logger.info("=" * 70)
    logger.info("TRAINING MODEL WITH HOST-BASED FEATURES ONLY (18 features)")
    logger.info("=" * 70)
    logger.info("⚠️  WARNING: This will be SLOW due to DNS/SSL lookups!")
    logger.info("=" * 70)
    
    # Load data
    logger.info(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, on_bad_lines='skip')
    
    if max_rows:
        logger.info(f"Limiting to first {max_rows} rows")
        df = df.head(max_rows)
    
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    
    logger.info(f"Loaded {len(df)} URLs")
    logger.info(f"  Legitimate (0): {len(df[df['label'] == 0])}")
    logger.info(f"  Phishing (1): {len(df[df['label'] == 1])}")
    
    urls = df["url"].tolist()
    labels = df["label"].tolist()
    
    # Extract features (host-based - DNS/SSL ENABLED)
    logger.info("\nExtracting HOST-BASED features only...")
    logger.info("⏳ This may take several hours for large datasets...")
    
    feature_extractor = FeatureExtractor(
        enable_dns_lookup=True,   # REQUIRED for host features
        enable_ssl_lookup=True,   # REQUIRED for host features
        enable_whois_lookup=False,
        timeout=5,
    )
    
    features_list = []
    valid_labels = []
    
    for i, (url, label) in enumerate(zip(urls, labels)):
        try:
            if i % 10 == 0:
                logger.info(f"Processing {i}/{len(urls)}... (slow due to network lookups)")
            features = feature_extractor.extract_features(url)
            features_list.append(features)
            valid_labels.append(int(label))
        except Exception as e:
            logger.warning(f"Failed to extract features from {url}: {e}")
    
    # Filter to host-based features only
    df_features = pd.DataFrame(features_list)
    available_host_features = [col for col in df_features.columns if col in HOST_FEATURES]
    df_host = df_features[available_host_features]
    
    logger.info(f"\nExtracted {len(available_host_features)} host-based features")
    
    if len(available_host_features) < len(HOST_FEATURES):
        missing = HOST_FEATURES - set(available_host_features)
        logger.warning(f"Missing features: {missing}")
    
    X = df_host.values
    y = np.array(valid_labels, dtype=np.int32)
    
    logger.info(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train model
    logger.info("\nTraining XGBoost model on host-based features...")
    model = PhishingModel(random_state=42)
    model.train(X, y, None, None, available_host_features)
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    logger.info(f"\nModel saved to {output_dir}")
    
    # Feature importance
    logger.info("\nTop 10 Most Important Host-Based Features:")
    feature_importance = model.get_feature_importance()
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        logger.info(f"{i}. {feature}: {importance:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info(f"Model location: {output_dir}")
    logger.info(f"Features used: {len(available_host_features)} host-based features")
    logger.info("\nTo evaluate this model, run:")
    logger.info(f"python evaluate_host_only.py <test_data.csv>")
    logger.info("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Train model using only host-based features")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data CSV file (url, label columns)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/host_only",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to use from CSV",
    )
    
    args = parser.parse_args()
    
    try:
        train_host_only(
            data_path=args.data,
            output_dir=args.output,
            max_rows=args.max_rows,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
