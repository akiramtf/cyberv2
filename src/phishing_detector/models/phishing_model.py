"""Single XGBoost model for phishing detection"""

import os
import joblib
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class PhishingModel:
    """Single XGBoost model for phishing detection"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def build_model(self) -> None:
        """Build the XGBoost model"""
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        logger.info("XGBoost model built successfully")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Train the model"""

        if feature_names is not None:
            self.feature_names = feature_names

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Train XGBoost
        logger.info("Training XGBoost model...")
        eval_set = [(X_train_scaled, y_train)]
        if X_val is not None:
            eval_set.append((X_val_scaled, y_val))

        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        self.is_trained = True
        logger.info("Model trained successfully")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        importance = self.model.feature_importances_

        if self.feature_names is not None:
            importance_dict = {
                name: float(imp) for name, imp in zip(self.feature_names, importance)
            }
            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
        else:
            importance_dict = {
                f"feature_{i}": float(imp) for i, imp in enumerate(importance)
            }

        return importance_dict

    def save(self, directory: str) -> None:
        """Save model and scaler"""
        os.makedirs(directory, exist_ok=True)

        # Save scaler
        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))

        # Save XGBoost
        self.model.save_model(os.path.join(directory, "xgboost_model.json"))

        # Save feature names
        if self.feature_names is not None:
            joblib.dump(self.feature_names, os.path.join(directory, "feature_names.pkl"))

        logger.info(f"Model saved to {directory}")

    def load(self, directory: str) -> None:
        """Load model and scaler"""
        # Load scaler
        self.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))

        # Load XGBoost
        self.model = xgb.XGBClassifier()
        self.model.load_model(os.path.join(directory, "xgboost_model.json"))

        # Load feature names
        feature_names_path = os.path.join(directory, "feature_names.pkl")
        if os.path.exists(feature_names_path):
            self.feature_names = joblib.load(feature_names_path)

        self.is_trained = True
        logger.info(f"Model loaded from {directory}")
