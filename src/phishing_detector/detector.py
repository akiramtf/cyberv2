"""Main phishing detector combining ensemble and zero-day detection"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from .features.extractor import FeatureExtractor
from .models.ensemble_classifier import EnsembleClassifier
from .models.zero_day_detector import ZeroDayDetector
from .legitimate_domains import LEGITIMATE_DOMAINS
import tldextract

logger = logging.getLogger(__name__)


class PhishingDetector:
    """Main phishing detection system"""

    def __init__(
        self,
        enable_zero_day: bool = True,
        enable_dns_lookup: bool = True,
        anomaly_threshold: float = 0.3,
        random_state: int = 42,
    ):
        """
        Initialize phishing detector

        Args:
            enable_zero_day: Enable zero-day anomaly detection
            enable_dns_lookup: Enable DNS lookups for host features
            anomaly_threshold: Threshold for zero-day detection (0-1)
            random_state: Random seed for reproducibility
        """
        self.enable_zero_day = enable_zero_day
        self.anomaly_threshold = anomaly_threshold

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            enable_dns_lookup=enable_dns_lookup,
            enable_whois_lookup=False,
            timeout=5,
        )

        # Initialize models
        self.ensemble_classifier = EnsembleClassifier(random_state=random_state)
        if enable_zero_day:
            self.zero_day_detector = ZeroDayDetector(
                contamination=anomaly_threshold, random_state=random_state
            )
        else:
            self.zero_day_detector = None

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

        # Check domain whitelist first (domain-based classification)
        # Extract domain (SLD + TLD) from URL using tldextract
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}".lower()  # e.g., "google.com"

        # If domain is in whitelist, immediately return SAFE
        if domain in LEGITIMATE_DOMAINS:
            return {
                "url": url,
                "is_phishing": False,
                "confidence": 0.99,  # Very high confidence for known legitimate domains
                "ensemble_score": 0.0,  # No ML ensemble used for whitelisted domains
                "prediction_source": "domain_whitelist",
                "zero_day_detected": False,
                "anomaly_score": 0.0,
                "risk_level": "safe",
                "model_scores": None,  # No ML models used for whitelisted domains
            }

        # Extract features
        features = self.feature_extractor.extract_features(url)
        feature_values = np.array([list(features.values())])

        # Get ensemble prediction
        ensemble_proba = self.ensemble_classifier.predict_proba(feature_values)[0]
        ensemble_prediction = int(ensemble_proba >= 0.5)

        # Get individual model scores
        individual_scores = self.ensemble_classifier.get_individual_predictions(feature_values)

        # Calculate confidence: for safe URLs, invert the probability
        # This makes confidence represent "how confident we are in the prediction"
        # rather than "probability of phishing"
        if ensemble_prediction == 1:  # Phishing
            confidence = ensemble_proba
        else:  # Safe
            confidence = 1.0 - ensemble_proba

        result = {
            "url": url,
            "is_phishing": bool(ensemble_prediction),
            "confidence": float(confidence),
            "ensemble_score": float(ensemble_proba),  # Raw ML ensemble score (phishing probability)
            "prediction_source": "ensemble",
            "zero_day_detected": False,
            "anomaly_score": 0.0,
            "model_scores": individual_scores,  # Add individual model scores
        }

        # Always calculate anomaly score if zero-day detector exists (for display)
        # But only use it for prediction if enable_zero_day is True
        if self.zero_day_detector is not None:
            anomaly_score = self.zero_day_detector.predict_anomaly_score(feature_values)[0]
            result["anomaly_score"] = float(anomaly_score)

            # Only use anomaly detection for prediction if enabled
            if self.enable_zero_day:
                is_anomaly = anomaly_score >= self.anomaly_threshold
                result["zero_day_detected"] = bool(is_anomaly)

                # If zero-day detected but ensemble says legitimate, flag as suspicious
                if is_anomaly and not ensemble_prediction:
                    result["is_phishing"] = True
                    result["prediction_source"] = "zero_day_detector"
                    result["confidence"] = float(anomaly_score)

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
    ) -> Dict[str, any]:
        """
        Train the detector

        Args:
            X_train: Training features
            y_train: Training labels (0 = legitimate, 1 = phishing)
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names

        Returns:
            Training history
        """
        logger.info("Training phishing detector...")

        # Train ensemble classifier
        history = self.ensemble_classifier.train(
            X_train, y_train, X_val, y_val, feature_names
        )

        # Train zero-day detector on legitimate URLs only
        if self.enable_zero_day and self.zero_day_detector is not None:
            X_legitimate = X_train[y_train == 0]
            self.zero_day_detector.train(X_legitimate)

        self.is_trained = True
        logger.info("Phishing detector trained successfully")

        return history

    def save(self, directory: str) -> None:
        """Save all models"""
        os.makedirs(directory, exist_ok=True)

        # Save ensemble classifier
        self.ensemble_classifier.save(directory)

        # Save zero-day detector
        if self.enable_zero_day and self.zero_day_detector is not None:
            self.zero_day_detector.save(directory)

        logger.info(f"Phishing detector saved to {directory}")

    def load(self, directory: str) -> None:
        """Load all models"""
        # Load ensemble classifier
        self.ensemble_classifier.load(directory)

        # Always load zero-day detector if it exists (for anomaly score display)
        # But only use it for predictions if enable_zero_day is True
        zero_day_scaler_path = os.path.join(directory, "zero_day_scaler.pkl")
        if os.path.exists(zero_day_scaler_path):
            if self.zero_day_detector is None:
                self.zero_day_detector = ZeroDayDetector()
            self.zero_day_detector.load(directory)
            logger.info("Zero-day detector loaded (anomaly scores will be displayed)")
        else:
            logger.warning("Zero-day detector not found, anomaly scores will show 0.0%")
            self.zero_day_detector = None

        self.is_trained = True
        logger.info(f"Phishing detector loaded from {directory}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble models"""
        return self.ensemble_classifier.get_feature_importance()
