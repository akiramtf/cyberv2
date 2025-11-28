"""Main phishing detector using XGBoost classification"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
import warnings

# Suppress urllib3 SSL warning
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from .features.extractor import FeatureExtractor
from .models.phishing_model import PhishingModel
from .models.hybrid_model import HybridPhishingModel
from .utils.url_tokenizer import URLTokenizer

logger = logging.getLogger(__name__)


class PhishingDetector:
    """Main phishing detection system"""

    def __init__(
        self,
        model_type: str = "hybrid",
        enable_dns_lookup: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize phishing detector

        Args:
            model_type: Type of model to use ("hybrid" or "xgboost")
            enable_dns_lookup: Enable DNS lookups for host features
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        
        # Initialize feature extractor
        # IMPORTANT: Hybrid model was trained with DNS/SSL lookups DISABLED.
        # We must match this configuration to avoid feature mismatch (0s vs real values).
        if self.model_type == "hybrid":
            use_dns_ssl = False
        else:
            use_dns_ssl = enable_dns_lookup

        self.feature_extractor = FeatureExtractor(
            enable_dns_lookup=use_dns_ssl,
            enable_ssl_lookup=use_dns_ssl,  # Couple SSL with DNS for now
            enable_whois_lookup=False,
            timeout=5,
        )

        # Initialize model
        if self.model_type == "hybrid":
            self.model = HybridPhishingModel()
            self.tokenizer = URLTokenizer()
        else:
            self.model = PhishingModel(random_state=random_state)
            self.tokenizer = None

        self.is_trained = False

    def predict(self, url: str) -> Dict[str, any]:
        """
        Predict if a URL is phishing

        Args:
            url: URL to analyze

        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")

        # Extract tabular features
        features = self.feature_extractor.extract_features(url)
        
        if self.model_type == "hybrid":
            # For Hybrid model:
            # 1. Prepare tabular features (DataFrame expected by scaler)
            import pandas as pd
            X_tabular = pd.DataFrame([features]).fillna(0).values
            
            # 2. Prepare text features
            X_text = self.tokenizer.transform([url])
            
            # 3. Predict
            phishing_proba = self.model.predict_proba(X_tabular, X_text)[0]
            
        else:
            # For XGBoost model:
            feature_values = np.array([list(features.values())])
            phishing_proba = self.model.predict_proba(feature_values)[0]
            
        is_phishing = int(phishing_proba >= 0.5)

        # Calculate confidence
        if is_phishing == 1:
            confidence = phishing_proba
        else:
            confidence = 1.0 - phishing_proba

        result = {
            "url": url,
            "is_phishing": bool(is_phishing),
            "phishing_score": float(phishing_proba),
            "prediction_source": self.model_type,
        }

        # Risk level based on confidence
        if result["is_phishing"]:
            # For phishing: higher confidence = higher risk
            if confidence >= 0.9:
                result["risk_level"] = "critical"
            elif confidence >= 0.7:
                result["risk_level"] = "high"
            else:
                result["risk_level"] = "medium"
        else:
            # For safe: higher confidence = lower risk
            if confidence >= 0.9:
                result["risk_level"] = "safe"
            elif confidence >= 0.7:
                result["risk_level"] = "low"
            else:
                result["risk_level"] = "medium"

        return result

    def predict_batch(self, urls: List[str]) -> List[Dict[str, any]]:
        """
        Predict multiple URLs

        Args:
            urls: List of URLs to analyze

        Returns:
            List of prediction results
        """
        results = []
        for url in urls:
            try:
                result = self.predict(url)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {url}: {e}")
                results.append(
                    {
                        "url": url,
                        "is_phishing": False,
                        "phishing_score": 0.0,
                        "prediction_source": self.model_type,
                        "error": str(e),
                        "risk_level": "unknown",
                    }
                )
        return results

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Train the detector
        
        Note: This method is primarily for the XGBoost model workflow.
        Hybrid model training is handled by train_hybrid_model.py
        """
        if self.model_type == "hybrid":
            logger.warning("Please use train_hybrid_model.py to train the Hybrid model.")
            return

        logger.info("Training phishing detector...")

        # Train model
        self.model.train(
            X_train, y_train, X_val, y_val, feature_names
        )

        self.is_trained = True
        logger.info("Phishing detector trained successfully")

    def save(self, directory: str) -> None:
        """Save all models"""
        os.makedirs(directory, exist_ok=True)

        # Save model
        self.model.save(directory)
        
        if self.model_type == "hybrid":
            self.tokenizer.save(os.path.join(directory, "tokenizer.json"))

        logger.info(f"Phishing detector saved to {directory}")

    def load(self, directory: str) -> None:
        """Load all models"""
        # Load model
        if self.model_type == "hybrid":
            # Hybrid model saves to a subdirectory usually, but let's assume standard path
            # If directory contains 'hybrid_v1', use it directly
            # Otherwise check if we need to append it
            
            # Try loading directly first
            try:
                self.model.load(directory)
                self.tokenizer.load(os.path.join(directory, "tokenizer.json"))
            except:
                # Fallback to hybrid_v1 subdirectory if standard load fails
                hybrid_dir = os.path.join(directory, "hybrid_v1")
                if os.path.exists(hybrid_dir):
                    self.model.load(hybrid_dir)
                    self.tokenizer.load(os.path.join(hybrid_dir, "tokenizer.json"))
                else:
                    raise
        else:
            self.model.load(directory)

        self.is_trained = True
        logger.info(f"Phishing detector loaded from {directory}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if self.model_type == "hybrid":
            return {} # Hybrid model doesn't support feature importance yet
        return self.model.get_feature_importance()
