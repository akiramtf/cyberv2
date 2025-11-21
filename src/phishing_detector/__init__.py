"""Phishing Detector Package"""

import warnings

# Suppress urllib3 SSL warning
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from .detector import PhishingDetector

__all__ = ["PhishingDetector"]
