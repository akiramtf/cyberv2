"""Simple usage example for the phishing detector"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phishing_detector.detector import PhishingDetector


def main():
    """Simple usage example"""

    print("=" * 60)
    print("CyberV2 - URL Phishing Detector - Simple Usage Example")
    print("=" * 60)

    # Initialize detector
    print("\n1. Initializing detector...")
    detector = PhishingDetector(
        enable_zero_day=True, enable_dns_lookup=False  # Disable DNS for faster demo
    )

    # Check if model exists
    model_path = "models/trained"
    try:
        detector.load(model_path)
        print(f"   ✓ Model loaded from {model_path}")
    except Exception as e:
        print(f"   ✗ Model not found. Please train the model first:")
        print(f"     python training/train.py")
        return

    # Test URLs
    test_urls = [
        "https://www.google.com",
        "https://www.github.com",
        "http://paypal-verify.suspicious-domain.tk/login.php",
        "http://192.168.1.1/bank/login",
        "https://www.wikipedia.org",
        "http://apple-id-locked.ml/verify",
    ]

    print(f"\n2. Testing {len(test_urls)} URLs...")
    print("-" * 60)

    # Predict each URL
    for i, url in enumerate(test_urls, 1):
        result = detector.predict(url)

        # Format output
        status = "🚨 PHISHING" if result["is_phishing"] else "✓ SAFE"
        confidence = result["confidence"] * 100
        risk = result["risk_level"].upper()

        print(f"\n{i}. {url}")
        print(f"   Status: {status}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Risk Level: {risk}")

        if result["zero_day_detected"]:
            print(f"   ⚠️  Zero-day pattern detected!")
            print(f"   Anomaly Score: {result['anomaly_score']:.2f}")

    # Batch prediction example
    print("\n" + "=" * 60)
    print("3. Batch Prediction Example")
    print("=" * 60)

    batch_results = detector.predict_batch(test_urls)

    phishing_count = sum(1 for r in batch_results if r["is_phishing"])
    safe_count = len(batch_results) - phishing_count

    print(f"\n   Total URLs analyzed: {len(batch_results)}")
    print(f"   🚨 Phishing detected: {phishing_count}")
    print(f"   ✓ Safe URLs: {safe_count}")

    # Feature importance
    print("\n" + "=" * 60)
    print("4. Top 5 Most Important Features")
    print("=" * 60)

    importance = detector.get_feature_importance()
    for i, (feature, score) in enumerate(list(importance.items())[:5], 1):
        print(f"   {i}. {feature}: {score:.4f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
