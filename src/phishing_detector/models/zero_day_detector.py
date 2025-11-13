"""Zero-day phishing detector using anomaly detection"""

import os
import joblib
import numpy as np
from typing import Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class ZeroDayDetector:
    """Detect zero-day phishing attacks using anomaly detection"""

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 150,
        max_samples: int = 256,
        random_state: int = 42,
    ):
        """
        Initialize zero-day detector

        Args:
            contamination: Expected proportion of outliers in dataset
            n_estimators: Number of base estimators in the ensemble
            max_samples: Number of samples to draw for each base estimator
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_trained = False

    def train(self, X_legitimate: np.ndarray) -> None:
        """
        Train anomaly detector on legitimate URLs only

        Args:
            X_legitimate: Feature matrix of legitimate URLs (label = 0)
        """
        logger.info("Training zero-day detector on legitimate URLs...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_legitimate)

        # Train isolation forest
        self.isolation_forest.fit(X_scaled)

        self.is_trained = True
        logger.info("Zero-day detector trained successfully")

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """
        Predict if URLs are anomalies (potential zero-day phishing)

        Args:
            X: Feature matrix

        Returns:
            Array of predictions (1 = anomaly/phishing, 0 = normal)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)

        # Predict (-1 = outlier/anomaly, 1 = inlier/normal)
        predictions = self.isolation_forest.predict(X_scaled)

        # Convert to binary (1 = anomaly, 0 = normal)
        anomaly_predictions = (predictions == -1).astype(int)

        return anomaly_predictions

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for URLs

        Args:
            X: Feature matrix

        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)

        # Get anomaly scores (lower = more anomalous)
        scores = self.isolation_forest.score_samples(X_scaled)

        # Normalize to [0, 1] range where higher = more anomalous
        # Using sigmoid transformation
        normalized_scores = 1 / (1 + np.exp(scores))

        return normalized_scores

    def set_threshold(self, threshold: float) -> None:
        """
        Set custom anomaly threshold

        Args:
            threshold: Threshold for anomaly detection (0-1, higher = more sensitive)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        # Adjust contamination parameter
        self.contamination = threshold
        self.isolation_forest.set_params(contamination=threshold)
        logger.info(f"Anomaly threshold set to {threshold}")

    def save(self, directory: str) -> None:
        """Save model and scaler"""
        os.makedirs(directory, exist_ok=True)

        # Save scaler
        joblib.dump(self.scaler, os.path.join(directory, "zero_day_scaler.pkl"))

        # Save isolation forest
        joblib.dump(
            self.isolation_forest, os.path.join(directory, "isolation_forest_model.pkl")
        )

        logger.info(f"Zero-day detector saved to {directory}")

    def load(self, directory: str) -> None:
        """Load model and scaler"""
        # Load scaler
        self.scaler = joblib.load(os.path.join(directory, "zero_day_scaler.pkl"))

        # Load isolation forest
        self.isolation_forest = joblib.load(
            os.path.join(directory, "isolation_forest_model.pkl")
        )

        self.is_trained = True
        logger.info(f"Zero-day detector loaded from {directory}")
