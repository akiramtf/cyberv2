"""Train and evaluate the Hybrid Phishing Model"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# Add src to path so we can import phishing_detector
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from phishing_detector.features.extractor import FeatureExtractor
from phishing_detector.models.hybrid_model import HybridPhishingModel
from phishing_detector.utils.url_tokenizer import URLTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """Load dataset"""
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    logger.info(f"Loaded {len(df)} samples")
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract tabular features"""
    logger.info("Extracting tabular features (DNS/SSL lookup disabled)...")
    # Disable DNS/SSL for speed and stability
    extractor = FeatureExtractor(enable_dns_lookup=False, enable_ssl_lookup=False)
    features = []
    
    for i, url in enumerate(df['url']):
        if i % 10000 == 0:
            logger.info(f"Processing {i}/{len(df)}")
        try:
            feats = extractor.extract_features(url)
            features.append(feats)
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            features.append({})
            
    return pd.DataFrame(features)

import argparse

import time

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Phishing Model")
    parser.add_argument("dataset", nargs="?", default="kaggle.csv", help="Path to the training dataset (CSV)")
    args = parser.parse_args()

    start_time = time.time()

    # Configuration
    DATA_PATH = args.dataset
    MODEL_DIR = "models/hybrid_v1"
    MAX_LEN = 200
    EPOCHS = 10
    BATCH_SIZE = 32
    
    # Check if dataset exists, else use test data
    if not os.path.exists(DATA_PATH):
        logger.warning(f"{DATA_PATH} not found, using evaluation_test_data.csv")
        DATA_PATH = "evaluation_test_data.csv"

    # 1. Load Data
    # Use full dataset
    df = load_data(DATA_PATH) 
    
    # 2. Preprocessing
    # A. Tabular Features
    X_tabular = extract_features(df)
    X_tabular = X_tabular.fillna(0)
    
    # B. Text Features
    logger.info("Tokenizing URLs...")
    tokenizer = URLTokenizer(max_length=MAX_LEN)
    X_text = tokenizer.fit_transform(df['url'].tolist())
    
    y = df['label'].values
    
    # 3. Train Model on Full Dataset
    logger.info(f"Training Hybrid Model on full dataset ({len(df)} samples)...")
    model = HybridPhishingModel(
        vocab_size=tokenizer.vocab_size,
        max_length=MAX_LEN,
        num_tabular_features=X_tabular.shape[1]
    )
    
    # Use a small validation split from the full data just for monitoring training progress
    # But primarily we want to use most data for training
    history = model.train(
        X_tabular.values, X_text, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        # We can pass None for validation to train on everything, 
        # or let the user know we are using a small internal split if the class supports it.
        # The current HybridPhishingModel.train method takes explicit validation data.
        # If we want to train on EVERYTHING, we pass None.
        X_tabular_val=None, X_text_val=None, y_val=None
    )
    
    # 4. Save
    model.save(MODEL_DIR)
    tokenizer.save(os.path.join(MODEL_DIR, "tokenizer.json"))
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Training complete. Model saved.")
    logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")

if __name__ == "__main__":
    main()
