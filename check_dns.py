"""Check DNS lookups and host-based features for a URL"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from src.phishing_detector.features.extractor import FeatureExtractor

def check_dns_for_url(url: str):
    """Check DNS and SSL features for a URL"""
    
    print("=" * 80)
    print(f"DNS & SSL LOOKUP FOR: {url}")
    print("=" * 80)
    
    # Create extractor with DNS/SSL enabled
    extractor = FeatureExtractor(
        enable_dns_lookup=True,
        enable_ssl_lookup=True,
        enable_whois_lookup=False,
        timeout=5
    )
    
    print("\n🔍 Extracting features (this may take a few seconds)...\n")
    
    # Extract all features
    features = extractor.extract_features(url)
    
    # Separate host-based features
    host_features = {
        k: v for k, v in features.items() 
        if any(x in k for x in ['dns', 'ssl', 'ptr', 'port'])
    }
    
    lexical_features = {
        k: v for k, v in features.items() 
        if k not in host_features
    }
    
    # Display DNS features
    print("📡 DNS FEATURES:")
    print("-" * 80)
    dns_features = {k: v for k, v in host_features.items() if 'dns' in k or 'ptr' in k}
    for name, value in dns_features.items():
        status = "✅" if value > 0 else "❌"
        print(f"  {status} {name:<30} = {value}")
    
    # Display SSL features
    print("\n🔒 SSL/TLS FEATURES:")
    print("-" * 80)
    ssl_features = {k: v for k, v in host_features.items() if 'ssl' in k}
    for name, value in ssl_features.items():
        if 'days' in name:
            print(f"  📅 {name:<30} = {value:.0f} days")
        elif value == 1.0:
            print(f"  ✅ {name:<30} = {value}")
        elif value == 0.0:
            print(f"  ❌ {name:<30} = {value}")
        else:
            print(f"  ℹ️  {name:<30} = {value}")
    
    # Display port features
    print("\n🔌 PORT FEATURES:")
    print("-" * 80)
    port_features = {k: v for k, v in host_features.items() if 'port' in k}
    for name, value in port_features.items():
        status = "✅" if value > 0 else "❌"
        print(f"  {status} {name:<30} = {value}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  Total features extracted: {len(features)}")
    print(f"  Lexical features: {len(lexical_features)}")
    print(f"  Host-based features: {len(host_features)}")
    print(f"    - DNS features: {len(dns_features)}")
    print(f"    - SSL features: {len(ssl_features)}")
    print(f"    - Port features: {len(port_features)}")
    print("=" * 80)


if __name__ == "__main__":
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "https://www.apple.com",
        "http://suspicious-site.tk",
    ]
    
    if len(sys.argv) > 1:
        # Use URL from command line
        url = sys.argv[1]
        check_dns_for_url(url)
    else:
        # Test with default URL
        print("Usage: python check_dns.py <url>")
        print("\nExample with default URL:\n")
        check_dns_for_url(test_urls[0])
        print("\n💡 Try: python check_dns.py https://www.example.com")
