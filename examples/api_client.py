"""Example API client for the phishing detector"""

import requests
import json


class PhishingDetectorClient:
    """Client for interacting with the phishing detector API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def health_check(self):
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def predict(self, url: str):
        """Predict if a single URL is phishing"""
        response = requests.post(f"{self.base_url}/predict", json={"url": url})
        return response.json()

    def batch_predict(self, urls: list):
        """Predict multiple URLs"""
        response = requests.post(f"{self.base_url}/batch_predict", json={"urls": urls})
        return response.json()

    def model_info(self):
        """Get model information"""
        response = requests.get(f"{self.base_url}/model/info")
        return response.json()


def main():
    """Example usage of the API client"""

    print("=" * 60)
    print("CyberV2 - API Client Example")
    print("=" * 60)

    # Initialize client
    client = PhishingDetectorClient()

    # Health check
    print("\n1. Health Check")
    print("-" * 60)
    try:
        health = client.health_check()
        print(json.dumps(health, indent=2))
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("   Please start the server first: python main.py")
        return

    # Single prediction
    print("\n2. Single URL Prediction")
    print("-" * 60)
    url = "http://paypal-verify.suspicious-domain.tk/login.php"
    print(f"Testing URL: {url}")

    try:
        result = client.predict(url)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")

    # Batch prediction
    print("\n3. Batch Prediction")
    print("-" * 60)
    urls = [
        "https://www.google.com",
        "http://suspicious-site.tk",
        "https://www.github.com",
        "http://phishing-example.ml/login",
    ]

    print(f"Testing {len(urls)} URLs...")

    try:
        results = client.batch_predict(urls)
        print(f"\nTotal URLs: {results['total_urls']}")
        print(f"Phishing detected: {results['phishing_count']}")
        print(f"Legitimate: {results['legitimate_count']}")

        print("\nDetailed Results:")
        for pred in results["predictions"]:
            status = "🚨 PHISHING" if pred["is_phishing"] else "✓ SAFE"
            print(f"  {pred['url'][:50]:50} -> {status} ({pred['confidence']:.2%})")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Model info
    print("\n4. Model Information")
    print("-" * 60)
    try:
        info = client.model_info()
        print(f"Model Version: {info['model_version']}")
        print(f"Features Count: {info['features_count']}")
        print(f"Zero-day Enabled: {info['zero_day_enabled']}")
        print("\nTop 5 Features:")
        for feat in info["top_features"][:5]:
            print(f"  - {feat['feature']}: {feat['importance']:.4f}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("API Client Demo Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
