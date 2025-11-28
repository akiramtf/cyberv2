"""Hybrid Deep Learning Model for Phishing Detection"""

import os
import json
import logging
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

class HybridPhishingModel:
    """
    Hybrid model combining:
    1. Tabular features (Dense layers)
    2. URL text features (CNN/LSTM)
    """
    
    def __init__(
        self, 
        vocab_size: int = 100, 
        max_length: int = 200, 
        num_tabular_features: int = 50,
        embedding_dim: int = 32
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_tabular_features = num_tabular_features
        self.embedding_dim = embedding_dim
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_model(self) -> None:
        """Build the Keras Functional API model"""
        
        # --- Branch A: Tabular Features ---
        input_tabular = layers.Input(shape=(self.num_tabular_features,), name="tabular_input")
        x_tab = layers.Dense(64, activation="relu")(input_tabular)
        x_tab = layers.BatchNormalization()(x_tab)
        x_tab = layers.Dropout(0.3)(x_tab)
        x_tab = layers.Dense(32, activation="relu")(x_tab)
        
        # --- Branch B: Text Features (URL) ---
        input_text = layers.Input(shape=(self.max_length,), name="text_input")
        x_text = layers.Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim)(input_text)
        
        # CNN layers for character-level patterns
        x_text = layers.Conv1D(64, kernel_size=3, activation="relu")(x_text)
        x_text = layers.MaxPooling1D(pool_size=2)(x_text)
        x_text = layers.Conv1D(128, kernel_size=3, activation="relu")(x_text)
        x_text = layers.GlobalMaxPooling1D()(x_text)
        x_text = layers.Dense(32, activation="relu")(x_text)
        
        # --- Combination ---
        combined = layers.concatenate([x_tab, x_text])
        
        z = layers.Dense(64, activation="relu")(combined)
        z = layers.Dropout(0.4)(z)
        z = layers.Dense(32, activation="relu")(z)
        output = layers.Dense(1, activation="sigmoid", name="output")(z)
        
        self.model = models.Model(inputs=[input_tabular, input_text], outputs=output)
        
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )
        
        logger.info("Hybrid Keras model built successfully")
        # self.model.summary(print_fn=logger.info)

    def train(
        self,
        X_tabular: np.ndarray,
        X_text: np.ndarray,
        y: np.ndarray,
        X_tabular_val: Optional[np.ndarray] = None,
        X_text_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 32
    ) -> Dict:
        """Train the model"""
        
        # Scale tabular features
        X_tabular_scaled = self.scaler.fit_transform(X_tabular)
        
        if self.model is None:
            self.build_model()
            
        validation_data = None
        if X_tabular_val is not None and X_text_val is not None and y_val is not None:
            X_tabular_val_scaled = self.scaler.transform(X_tabular_val)
            validation_data = ([X_tabular_val_scaled, X_text_val], y_val)
            
        monitor = "val_loss" if validation_data is not None else "loss"
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3)
        ]
        
        history = self.model.fit(
            [X_tabular_scaled, X_text],
            y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2
        )
        
        self.is_trained = True
        return history.history

    def predict_proba(self, X_tabular: np.ndarray, X_text: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        X_tabular_scaled = self.scaler.transform(X_tabular)
        return self.model.predict([X_tabular_scaled, X_text]).flatten()

    def predict(self, X_tabular: np.ndarray, X_text: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probs = self.predict_proba(X_tabular, X_text)
        return (probs > 0.5).astype(int)

    def save(self, directory: str) -> None:
        """Save model and scaler"""
        os.makedirs(directory, exist_ok=True)
        
        # Save Keras model
        self.model.save(os.path.join(directory, "hybrid_model.keras"))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))
        
        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "num_tabular_features": self.num_tabular_features,
            "embedding_dim": self.embedding_dim
        }
        with open(os.path.join(directory, "config.json"), 'w') as f:
            json.dump(config, f)
            
        logger.info(f"Model saved to {directory}")

    def load(self, directory: str) -> None:
        """Load model and scaler"""
        # Load config
        with open(os.path.join(directory, "config.json"), 'r') as f:
            config = json.load(f)
            
        self.vocab_size = config["vocab_size"]
        self.max_length = config["max_length"]
        self.num_tabular_features = config["num_tabular_features"]
        self.embedding_dim = config["embedding_dim"]
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))
        
        # Load Keras model
        self.model = models.load_model(os.path.join(directory, "hybrid_model.keras"))
        
        self.is_trained = True
        logger.info(f"Model loaded from {directory}")
