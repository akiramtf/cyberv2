"""Simple test of the phishing detector without API"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phishing_detector.detector import PhishingDetector


def main():
    print("Loading phishing detector...")

    try:
        # Initialize and load detector
        detector = PhishingDetector(enable_zero_day=True, enable_dns_lookup=False)
        detector.load("models/trained")
        print("✓ Model loaded successfully\n")

        # Test a few URLs
        test_urls = [
            "https://www.google.com",
            "http://paypal-verify.suspicious-domain.tk/login.php",
            "https://www.github.com",
        ]

        print("Testing URLs:\n")
        for url in test_urls:
            result = detector.predict(url)
            status = "🚨 PHISHING" if result["is_phishing"] else "✓ SAFE"
            confidence = result["confidence"] * 100
            print(f"{status} - {url}")
            print(f"  Confidence: {confidence:.1f}%")
            print(f"  Risk Level: {result['risk_level']}\n")

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
