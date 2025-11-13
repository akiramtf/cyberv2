"""ML models for phishing detection"""

from .ensemble_classifier import EnsembleClassifier
from .zero_day_detector import ZeroDayDetector

__all__ = ["EnsembleClassifier", "ZeroDayDetector"]
