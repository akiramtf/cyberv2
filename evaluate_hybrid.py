"""Evaluate Hybrid Model on Custom Data"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import argparse

# Add src to path so we can import phishing_detector
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from phishing_detector.features.extractor import FeatureExtractor
from phishing_detector.models.hybrid_model import HybridPhishingModel
from phishing_detector.utils.url_tokenizer import URLTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_custom_data(data_path: str, model_dir: str):
    """Evaluate hybrid model on custom data"""
    
    # 1. Load Data
    if not os.path.exists(data_path):
        logger.error(f"File not found: {data_path}")
        return

    logger.info(f"Loading custom data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Check if 'url' column exists
    if 'url' not in df.columns:
        logger.error("Dataset must contain a 'url' column")
        return

    # 2. Load Model & Tokenizer
    logger.info(f"Loading model from {model_dir}...")
    model = HybridPhishingModel()
    tokenizer = URLTokenizer()
    
    try:
        model.load(model_dir)
        tokenizer.load(os.path.join(model_dir, "tokenizer.json"))
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        return

    # 3. Preprocessing
    # A. Tabular Features
    logger.info("Extracting tabular features (DNS/SSL lookup disabled)...")
    extractor = FeatureExtractor(enable_dns_lookup=False, enable_ssl_lookup=False)
    features = []
    
    for i, url in enumerate(df['url']):
        if i % 1000 == 0:
            logger.info(f"Processing {i}/{len(df)}")
        try:
            feats = extractor.extract_features(url)
            features.append(feats)
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            features.append({})
            
    X_tabular = pd.DataFrame(features)
    X_tabular = X_tabular.fillna(0)
    
    # B. Text Features
    logger.info("Tokenizing URLs...")
    X_text = tokenizer.transform(df['url'].tolist())
    
    # 4. Predict
    logger.info("Predicting...")
    y_pred_proba = model.predict_proba(X_tabular.values, X_text)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 5. Evaluate (if labels exist)
    if 'label' in df.columns:
        y_true = df['label'].values
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0.0
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print("\n" + "="*60)
        print("DETAILED EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:            {acc:.4f}")
        print(f"Precision:           {prec:.4f}")
        print(f"Recall (Sensitivity):{rec:.4f}")
        print(f"F1-Score:            {f1:.4f}")
        print(f"ROC-AUC:             {auc:.4f}")
        print("-" * 30)
        print(f"False Positive Rate: {fpr:.4f} (LOWER IS BETTER)")
        print("-" * 30)
        print()
        print("Confusion Matrix:")
        print(f"True Negatives (Safe correctly identified):      {tn}")
        print(f"False Positives (Safe wrongly blocked):          {fp}")
        print(f"False Negatives (Phishing missed):               {fn}")
        print(f"True Positives (Phishing correctly blocked):     {tp}")
        print("="*60 + "\n")
        
        # Assign predictions for analysis
        df['predicted_label'] = y_pred
        df['phishing_probability'] = y_pred_proba
        
        # Error Analysis
        misclassified = df[df['label'] != df['predicted_label']].copy()
        misclassified_count = len(misclassified)
        
        if misclassified_count > 0:
            misclassified_path = data_path.replace(".csv", "_misclassified.csv")
            misclassified.to_csv(misclassified_path, index=False)
            
            print(f"⚠️  Found {misclassified_count} misclassified URLs.")
            print(f"   Details saved to: {misclassified_path}\n")
            
            # False Positives (Safe sites blocked) - True: 0, Pred: 1
            # Sort by probability descending (highest confidence it was phishing)
            fps = misclassified[(misclassified['label'] == 0) & (misclassified['predicted_label'] == 1)]
            fps = fps.sort_values('phishing_probability', ascending=False).head(5)
            
            print("Top 5 False Positives (Safe sites blocked):")
            if not fps.empty:
                for _, row in fps.iterrows():
                    print(f"   - {row['url']} (Score: {row['phishing_probability']:.2f})")
            else:
                print("   None")
            print()

            # False Negatives (Phishing sites missed) - True: 1, Pred: 0
            # Sort by probability ascending (lowest probability of phishing -> highest confidence it was safe)
            fns = misclassified[(misclassified['label'] == 1) & (misclassified['predicted_label'] == 0)]
            fns = fns.sort_values('phishing_probability', ascending=True).head(5)
            
            print("Top 5 False Negatives (Phishing sites missed):")
            if not fns.empty:
                for _, row in fns.iterrows():
                    print(f"   - {row['url']} (Score: {row['phishing_probability']:.2f})")
            else:
                print("   None")
            print("="*60 + "\n")
        
        # Save results
        output_path = data_path.replace(".csv", "_hybrid_results.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
    else:
        logger.info("No 'label' column found. Saving predictions only.")
        df['predicted_label'] = y_pred
        df['phishing_probability'] = y_pred_proba
        output_path = data_path.replace(".csv", "_hybrid_predictions.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Model on Custom Data")
    parser.add_argument("data_path", help="Path to the custom CSV file (must contain 'url' column)")
    parser.add_argument("--model_dir", default="models/hybrid_v1", help="Directory containing the trained model")
    
    args = parser.parse_args()
    
    evaluate_custom_data(args.data_path, args.model_dir)
