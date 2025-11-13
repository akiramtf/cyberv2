"""Ensemble classifier combining multiple ML models"""

import os
import joblib
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)


class EnsembleClassifier:
    """Ensemble classifier for phishing detection"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.rf_model = None
        self.nn_model = None
        self.voting_clf = None
        self.feature_names = None
        self.is_trained = False

    def build_models(self, n_features: int) -> None:
        """Build all models in the ensemble"""

        # XGBoost Classifier
        self.xgb_model = xgb.XGBClassifier(
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

        # Random Forest Classifier
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Neural Network
        self.nn_model = self._build_neural_network(n_features)

        logger.info("All models built successfully")

    def _build_neural_network(self, n_features: int) -> keras.Model:
        """Build neural network model"""
        model = keras.Sequential(
            [
                layers.Input(shape=(n_features,)),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        )

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """Train all models in the ensemble"""

        if feature_names is not None:
            self.feature_names = feature_names

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # Build models if not already built
        if self.xgb_model is None:
            self.build_models(X_train.shape[1])

        # Train XGBoost
        logger.info("Training XGBoost model...")
        eval_set = [(X_train_scaled, y_train)]
        if X_val is not None:
            eval_set.append((X_val_scaled, y_val))

        self.xgb_model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Train Random Forest
        logger.info("Training Random Forest model...")
        self.rf_model.fit(X_train_scaled, y_train)

        # Train Neural Network
        logger.info("Training Neural Network model...")
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=10,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        validation_data = (X_val_scaled, y_val) if X_val is not None else None

        history = self.nn_model.fit(
            X_train_scaled,
            y_train,
            validation_data=validation_data,
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=0,
        )

        self.is_trained = True
        logger.info("All models trained successfully")

        return {"nn_history": history.history}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble voting"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")

        X_scaled = self.scaler.transform(X)

        # Get predictions from each model
        xgb_pred = self.xgb_model.predict(X_scaled)
        rf_pred = self.rf_model.predict(X_scaled)
        nn_pred = (self.nn_model.predict(X_scaled, verbose=0) > 0.5).astype(int).flatten()

        # Weighted voting (XGBoost gets higher weight)
        weights = np.array([0.4, 0.3, 0.3])  # XGB, RF, NN
        predictions = np.column_stack([xgb_pred, rf_pred, nn_pred])
        weighted_votes = np.average(predictions, axis=1, weights=weights)

        # Final prediction (threshold at 0.5)
        final_predictions = (weighted_votes >= 0.5).astype(int)

        return final_predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using ensemble averaging"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")

        X_scaled = self.scaler.transform(X)

        # Get probability predictions from each model
        xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        nn_proba = self.nn_model.predict(X_scaled, verbose=0).flatten()

        # Weighted averaging
        weights = np.array([0.4, 0.3, 0.3])  # XGB, RF, NN
        probabilities = np.column_stack([xgb_proba, rf_proba, nn_proba])
        weighted_proba = np.average(probabilities, axis=1, weights=weights)

        return weighted_proba

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from tree-based models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before getting feature importance")

        # Get importance from XGBoost and Random Forest
        xgb_importance = self.xgb_model.feature_importances_
        rf_importance = self.rf_model.feature_importances_

        # Average importance
        avg_importance = (xgb_importance + rf_importance) / 2

        if self.feature_names is not None:
            importance_dict = {
                name: float(imp) for name, imp in zip(self.feature_names, avg_importance)
            }
            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
        else:
            importance_dict = {
                f"feature_{i}": float(imp) for i, imp in enumerate(avg_importance)
            }

        return importance_dict

    def save(self, directory: str) -> None:
        """Save all models and scaler"""
        os.makedirs(directory, exist_ok=True)

        # Save scaler
        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))

        # Save XGBoost
        self.xgb_model.save_model(os.path.join(directory, "xgboost_model.json"))

        # Save Random Forest
        joblib.dump(self.rf_model, os.path.join(directory, "random_forest_model.pkl"))

        # Save Neural Network
        self.nn_model.save(os.path.join(directory, "neural_network_model.h5"))

        # Save feature names
        if self.feature_names is not None:
            joblib.dump(self.feature_names, os.path.join(directory, "feature_names.pkl"))

        logger.info(f"Models saved to {directory}")

    def load(self, directory: str) -> None:
        """Load all models and scaler"""
        # Load scaler
        self.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))

        # Load XGBoost
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(os.path.join(directory, "xgboost_model.json"))

        # Load Random Forest
        self.rf_model = joblib.load(os.path.join(directory, "random_forest_model.pkl"))

        # Load Neural Network
        self.nn_model = keras.models.load_model(
            os.path.join(directory, "neural_network_model.h5")
        )

        # Load feature names
        feature_names_path = os.path.join(directory, "feature_names.pkl")
        if os.path.exists(feature_names_path):
            self.feature_names = joblib.load(feature_names_path)

        self.is_trained = True
        logger.info(f"Models loaded from {directory}")
