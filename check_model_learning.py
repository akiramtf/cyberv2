"""Check model's feature importance and predictions"""

from src.phishing_detector.detector import PhishingDetector

# Load detector
detector = PhishingDetector(enable_zero_day=False, enable_dns_lookup=False)
detector.load("models/trained")

print("=" * 70)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 70)

# Get feature importance
importance = detector.get_feature_importance()
for i, (feature, score) in enumerate(list(importance.items())[:15], 1):
    print(f"{i:2d}. {feature:35s} {score:.4f}")

print("\n" + "=" * 70)
print("DETAILED PREDICTIONS FOR SUSPICIOUS URLs")
print("=" * 70)

# Test specific suspicious URLs
test_cases = [
    ("http://suspicious-site.tk", "Should be PHISHING (.tk TLD)"),
    ("http://jo-onlline-islamic.com", "Should be PHISHING (typo)"),
    ("http://paypal-verify.suspicious-domain.tk/login.php", "Should be PHISHING"),
]

for url, expected in test_cases:
    result = detector.predict(url)
    status = "🚨 PHISHING" if result["is_phishing"] else "✓ SAFE"
    match = "✅" if ("PHISHING" in expected and result["is_phishing"]) else "❌"

    print(f"\n{match} {status} - {url}")
    print(f"   Expected: {expected}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Risk Level: {result['risk_level']}")
