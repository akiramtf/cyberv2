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

logger = logging.getLogger(__name__)


class PhishingDetector:
    """Main phishing detection system"""

    def __init__(
        self,
        enable_dns_lookup: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize phishing detector

        Args:
            enable_dns_lookup: Enable DNS lookups for host features
            random_state: Random seed for reproducibility
        """
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            enable_dns_lookup=enable_dns_lookup,
            enable_ssl_lookup=enable_dns_lookup,  # Couple SSL with DNS for now
            enable_whois_lookup=False,
            timeout=5,
        )

        # Initialize model
        self.model = PhishingModel(random_state=random_state)

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



        # Extract features
        features = self.feature_extractor.extract_features(url)
        feature_values = np.array([list(features.values())])

        # Get prediction
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
            "confidence": float(confidence),
            "phishing_score": float(phishing_proba),
            "prediction_source": "model",
        }

        # Risk level based on confidence
        if result["is_phishing"]:
            # For phishing: higher confidence = higher risk
            if result["confidence"] >= 0.9:
                result["risk_level"] = "critical"
            elif result["confidence"] >= 0.7:
                result["risk_level"] = "high"
            else:
                result["risk_level"] = "medium"
        else:
            # For safe: higher confidence = lower risk
            if result["confidence"] >= 0.9:
                result["risk_level"] = "safe"
            elif result["confidence"] >= 0.7:
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
                        "confidence": 0.0,
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

        Args:
            X_train: Training features
            y_train: Training labels (0 = legitimate, 1 = phishing)
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
        """
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

        logger.info(f"Phishing detector saved to {directory}")

    def load(self, directory: str) -> None:
        """Load all models"""
        # Load model
        self.model.load(directory)

        self.is_trained = True
        logger.info(f"Phishing detector loaded from {directory}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        return self.model.get_feature_importance()
