#!/usr/bin/env python3
"""
Simple script to train phishing detection models from a CSV file.

CSV file should have two columns:
- url: The URL to analyze
- label: 0 for legitimate, 1 for phishing

Usage:
    python train_from_csv.py data.csv
    python train_from_csv.py data.csv --test-split 0.2
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBClassifier

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from phishing_detector.features.extractor import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(csv_file: str) -> pd.DataFrame:
    """Load training data from CSV file"""
    logger.info(f"Loading data from: {csv_file}")

    # Try different parsing strategies
    try:
        # First try: standard CSV with proper quoting
        df = pd.read_csv(csv_file, quoting=1, on_bad_lines='skip')
    except Exception as e:
        logger.warning(f"Standard parsing failed: {e}")
        try:
            # Second try: read all lines manually
            logger.info("Trying manual parsing...")
            df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='utf-8', engine='python')
        except Exception as e2:
            logger.warning(f"Python engine failed: {e2}")
            # Third try: read with error skipping
            logger.info("Trying with error handling...")
            df = pd.read_csv(csv_file, on_bad_lines='warn', encoding='utf-8')

    # Verify columns
    if 'url' not in df.columns or 'label' not in df.columns:
        raise ValueError(
            f"CSV must have 'url' and 'label' columns. Found columns: {list(df.columns)}"
        )

    # Clean data
    df = df[['url', 'label']].copy()
    df = df.dropna()

    # Ensure label is int
    df['label'] = df['label'].astype(int)

    # Filter valid labels
    df = df[df['label'].isin([0, 1])]

    logger.info(f"Loaded {len(df)} URLs")
    logger.info(f"  Legitimate: {len(df[df['label'] == 0])}")
    logger.info(f"  Phishing: {len(df[df['label'] == 1])}")

    return df


def extract_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from URLs"""
    logger.info("Extracting features from URLs...")

    extractor = FeatureExtractor()
    features_list = []
    labels_list = []

    for idx, row in df.iterrows():
        try:
            features = extractor.extract(row['url'])
            features_list.append(list(features.values()))
            labels_list.append(row['label'])

            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed {idx + 1}/{len(df)} URLs")
        except Exception as e:
            logger.warning(f"  Failed to extract features for {row['url']}: {e}")
            continue

    X = np.array(features_list)
    y = np.array(labels_list)

    logger.info(f"Extracted features: {X.shape}")
    return X, y


def train_models(X_train, y_train, X_test, y_test):
    """Train all three models"""

    # Scale features
    logger.info("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}

    # 1. Train XGBoost
    logger.info("\n" + "="*60)
    logger.info("Training XGBoost classifier...")
    logger.info("="*60)

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train)

    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)

    logger.info(f"\nXGBoost Accuracy: {xgb_accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, xgb_pred,
                                             target_names=['Legitimate', 'Phishing']))

    models['xgboost'] = xgb_model

    # 2. Train Random Forest
    logger.info("\n" + "="*60)
    logger.info("Training Random Forest classifier...")
    logger.info("="*60)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)

    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    logger.info(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, rf_pred,
                                             target_names=['Legitimate', 'Phishing']))

    models['random_forest'] = rf_model

    # 3. Train Neural Network
    logger.info("\n" + "="*60)
    logger.info("Training Neural Network...")
    logger.info("="*60)

    nn_model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = nn_model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    nn_pred_proba = nn_model.predict(X_test_scaled, verbose=0)
    nn_pred = (nn_pred_proba > 0.5).astype(int).flatten()
    nn_accuracy = accuracy_score(y_test, nn_pred)

    logger.info(f"\nNeural Network Accuracy: {nn_accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, nn_pred,
                                             target_names=['Legitimate', 'Phishing']))

    models['neural_network'] = nn_model

    # Ensemble predictions
    logger.info("\n" + "="*60)
    logger.info("Ensemble Performance (Weighted Voting)")
    logger.info("="*60)

    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    nn_proba = nn_pred_proba.flatten()

    # Weighted voting: XGBoost 40%, RF 30%, NN 30%
    ensemble_proba = (0.4 * xgb_proba + 0.3 * rf_proba + 0.3 * nn_proba)
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

    logger.info(f"\nEnsemble Accuracy: {ensemble_accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, ensemble_pred,
                                             target_names=['Legitimate', 'Phishing']))

    logger.info("\nConfusion Matrix:")
    logger.info(str(confusion_matrix(y_test, ensemble_pred)))

    return models, scaler


def save_models(models: dict, scaler, output_dir: str = "models"):
    """Save trained models"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving models to {output_path}/")

    # Save sklearn models
    joblib.dump(models['xgboost'], output_path / 'xgboost_model.pkl')
    logger.info("  ✓ Saved xgboost_model.pkl")

    joblib.dump(models['random_forest'], output_path / 'random_forest_model.pkl')
    logger.info("  ✓ Saved random_forest_model.pkl")

    # Save neural network
    models['neural_network'].save(output_path / 'neural_network_model.keras')
    logger.info("  ✓ Saved neural_network_model.keras")

    # Save scaler
    joblib.dump(scaler, output_path / 'scaler.pkl')
    logger.info("  ✓ Saved scaler.pkl")

    logger.info("\n✅ All models saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Train phishing detection models from CSV file"
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to CSV file with url and label columns'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Test set split ratio (default: 0.2)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for models (default: models)'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("PHISHING DETECTOR MODEL TRAINING")
    logger.info("="*60)

    # Load data
    df = load_data(args.csv_file)

    # Extract features
    X, y = extract_features(df)

    # Split data
    logger.info(f"\nSplitting data (test size: {args.test_split})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )
    logger.info(f"  Training set: {len(X_train)} samples")
    logger.info(f"  Test set: {len(X_test)} samples")

    # Train models
    models, scaler = train_models(X_train, y_train, X_test, y_test)

    # Save models
    save_models(models, scaler, args.output_dir)

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info("\nTo use the models, start the API server:")
    logger.info("  python src/phishing_detector/api/service.py")
    logger.info("\nThen open ui.html in your browser to test URLs")


if __name__ == "__main__":
    main()
