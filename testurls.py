from src.phishing_detector.detector import PhishingDetector

# Load the trained model
detector = PhishingDetector(enable_zero_day=True, enable_dns_lookup=False)
detector.load("models/trained")

# Test your URLs here
my_urls = [
    "https://www.google.com",
    "http://suspicious-site.tk",
    "https://yourbank.com/login",
    "https://www.fastwork.com",      # Added comma
    "https://www.apple.com",          # Added comma
    "https://www.soodaza.com",        # Added comma
    "http://jo-onlline-islamic.com",  # Added comma
]

for url in my_urls:
    result = detector.predict(url)
    status = "🚨 PHISHING" if result["is_phishing"] else "✓ SAFE"
    print(f"\n{status} - {url}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Risk Level: {result['risk_level']}")
    if result['zero_day_detected']:
        print(f"  ⚠️ Zero-day anomaly detected!")