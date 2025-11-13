"""Unit tests for ML models"""

import pytest
import numpy as np
from src.phishing_detector.models.ensemble_classifier import EnsembleClassifier
from src.phishing_detector.models.zero_day_detector import ZeroDayDetector


class TestEnsembleClassifier:
    """Test ensemble classifier"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 20)
        y_test = np.random.randint(0, 2, 20)
        return X_train, y_train, X_test, y_test

    def test_build_models(self):
        """Test model building"""
        classifier = EnsembleClassifier(random_state=42)
        classifier.build_models(n_features=20)

        assert classifier.xgb_model is not None
        assert classifier.rf_model is not None
        assert classifier.nn_model is not None

    def test_train_and_predict(self, sample_data):
        """Test training and prediction"""
        X_train, y_train, X_test, y_test = sample_data

        classifier = EnsembleClassifier(random_state=42)
        classifier.train(X_train, y_train)

        # Test prediction
        predictions = classifier.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, sample_data):
        """Test probability prediction"""
        X_train, y_train, X_test, y_test = sample_data

        classifier = EnsembleClassifier(random_state=42)
        classifier.train(X_train, y_train)

        probabilities = classifier.predict_proba(X_test)

        assert len(probabilities) == len(X_test)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_get_feature_importance(self, sample_data):
        """Test feature importance extraction"""
        X_train, y_train, X_test, y_test = sample_data
        feature_names = [f"feature_{i}" for i in range(20)]

        classifier = EnsembleClassifier(random_state=42)
        classifier.train(X_train, y_train, feature_names=feature_names)

        importance = classifier.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 20

    def test_predict_without_training(self, sample_data):
        """Test prediction without training raises error"""
        X_train, y_train, X_test, y_test = sample_data

        classifier = EnsembleClassifier(random_state=42)

        with pytest.raises(ValueError):
            classifier.predict(X_test)


class TestZeroDayDetector:
    """Test zero-day detector"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data"""
        np.random.seed(42)
        # Legitimate URLs (normal distribution)
        X_legitimate = np.random.randn(100, 20)
        # Anomalous URLs (shifted distribution)
        X_anomalous = np.random.randn(20, 20) + 3
        return X_legitimate, X_anomalous

    def test_train(self, sample_data):
        """Test training"""
        X_legitimate, _ = sample_data

        detector = ZeroDayDetector(random_state=42)
        detector.train(X_legitimate)

        assert detector.is_trained is True

    def test_predict_anomaly(self, sample_data):
        """Test anomaly prediction"""
        X_legitimate, X_anomalous = sample_data

        detector = ZeroDayDetector(contamination=0.1, random_state=42)
        detector.train(X_legitimate)

        # Predict on anomalous data
        predictions = detector.predict_anomaly(X_anomalous)

        assert len(predictions) == len(X_anomalous)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_anomaly_score(self, sample_data):
        """Test anomaly score prediction"""
        X_legitimate, X_anomalous = sample_data

        detector = ZeroDayDetector(random_state=42)
        detector.train(X_legitimate)

        # Get anomaly scores
        scores = detector.predict_anomaly_score(X_anomalous)

        assert len(scores) == len(X_anomalous)
        assert all(0 <= s <= 1 for s in scores)

    def test_set_threshold(self):
        """Test threshold setting"""
        detector = ZeroDayDetector()

        detector.set_threshold(0.2)
        assert detector.contamination == 0.2

        # Invalid threshold
        with pytest.raises(ValueError):
            detector.set_threshold(1.5)

    def test_predict_without_training(self, sample_data):
        """Test prediction without training raises error"""
        _, X_anomalous = sample_data

        detector = ZeroDayDetector()

        with pytest.raises(ValueError):
            detector.predict_anomaly(X_anomalous)
