from src.phishing_detector.detector import PhishingDetector

# Load the trained model WITHOUT zero-day detection (to avoid false positives)
detector = PhishingDetector(
    enable_zero_day=False,  # DISABLED - causes false positives with small training data
    enable_dns_lookup=False
)
detector.load("models/trained")

# Test your URLs here
my_urls = [
    "https://www.google.com",
    "http://suspicious-site.tk",
    "https://yourbank.com/login",
    "https://www.fastwork.com",
    "https://www.apple.com",
    "https://www.soodaza.com",
    "http://jo-onlline-islamic.com",
]

print("=" * 60)
print("URL Phishing Detection Results")
print("=" * 60)

for url in my_urls:
    result = detector.predict(url)
    status = "🚨 PHISHING" if result["is_phishing"] else "✓ SAFE"
    print(f"\n{status} - {url}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Risk Level: {result['risk_level']}")

print("\n" + "=" * 60)
