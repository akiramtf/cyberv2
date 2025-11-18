#!/usr/bin/env python3
"""
Analyze Dataset Difficulty
Checks if phishing URLs have obvious patterns that make classification "too easy"
"""

import pandas as pd
import re
from collections import Counter

def analyze_dataset_difficulty(csv_path, sample_size=100):
    """Analyze how "obvious" the phishing patterns are"""

    print(f"\n{'='*80}")
    print(f"DATASET DIFFICULTY ANALYSIS: {csv_path}")
    print(f"{'='*80}\n")

    # Load dataset
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df = df.dropna(subset=['url', 'label'])

    # Sample URLs
    legit = df[df['label'] == 0].head(sample_size)
    phishing = df[df['label'] == 1].head(sample_size)

    print(f"Analyzing {len(legit)} legitimate + {len(phishing)} phishing URLs\n")

    # Analyze phishing URLs for obvious patterns
    print("="*80)
    print("PHISHING URL ANALYSIS")
    print("="*80)

    phishing_urls = phishing['url'].tolist()

    # Check for IP addresses
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    ip_count = sum(1 for url in phishing_urls if re.search(ip_pattern, url))

    # Check for suspicious TLDs
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click',
                       '.link', '.bid', '.racing', '.download', '.date', '.accountant']
    tld_count = sum(1 for url in phishing_urls
                    for tld in suspicious_tlds if url.endswith(tld) or tld in url)

    # Check for common typos
    typo_patterns = ['gooogle', 'paypa1', 'faceboook', 'amaz0n', 'app1e', 'micros0ft']
    typo_count = sum(1 for url in phishing_urls
                     for typo in typo_patterns if typo.lower() in url.lower())

    # Check for suspicious keywords
    suspicious_keywords = ['verify', 'secure', 'login', 'update', 'suspended',
                          'confirm', 'account', 'signin', 'locked', 'alert']
    keyword_count = sum(1 for url in phishing_urls
                       for keyword in suspicious_keywords if keyword in url.lower())

    # Check for excessive hyphens
    hyphen_count = sum(1 for url in phishing_urls if url.count('-') > 2)

    # Check for @ symbol (rare but super obvious)
    at_count = sum(1 for url in phishing_urls if '@' in url)

    print(f"\n🚩 Obvious Phishing Indicators:")
    print(f"   IP Addresses:          {ip_count:4d} / {len(phishing_urls)} ({ip_count/len(phishing_urls)*100:.1f}%)")
    print(f"   Suspicious TLDs:       {tld_count:4d} / {len(phishing_urls)} ({tld_count/len(phishing_urls)*100:.1f}%)")
    print(f"   Common Typos:          {typo_count:4d} / {len(phishing_urls)} ({typo_count/len(phishing_urls)*100:.1f}%)")
    print(f"   Suspicious Keywords:   {keyword_count:4d} / {len(phishing_urls)} ({keyword_count/len(phishing_urls)*100:.1f}%)")
    print(f"   Excessive Hyphens:     {hyphen_count:4d} / {len(phishing_urls)} ({hyphen_count/len(phishing_urls)*100:.1f}%)")
    print(f"   @ Symbol:              {at_count:4d} / {len(phishing_urls)} ({at_count/len(phishing_urls)*100:.1f}%)")

    # Count URLs with at least one obvious indicator
    obvious_count = 0
    for url in phishing_urls:
        has_indicator = (
            re.search(ip_pattern, url) or
            any(tld in url for tld in suspicious_tlds) or
            any(typo in url.lower() for typo in typo_patterns) or
            any(keyword in url.lower() for keyword in suspicious_keywords) or
            url.count('-') > 2 or
            '@' in url
        )
        if has_indicator:
            obvious_count += 1

    print(f"\n{'='*80}")
    print(f"TOTAL with obvious indicators: {obvious_count} / {len(phishing_urls)} ({obvious_count/len(phishing_urls)*100:.1f}%)")
    print(f"{'='*80}")

    if obvious_count / len(phishing_urls) > 0.8:
        print("\n⚠️  DATASET IS TOO EASY!")
        print("   > 80% of phishing URLs have obvious patterns")
        print("   → Any basic model will get 99%+ accuracy")
        print("   → Different algorithms will produce similar results")
    elif obvious_count / len(phishing_urls) > 0.6:
        print("\n⚠️  DATASET IS MODERATELY EASY")
        print("   60-80% of phishing URLs have obvious patterns")
        print("   → Good models will get 95-98% accuracy")
    else:
        print("\n✅ DATASET HAS CHALLENGING PATTERNS")
        print("   < 60% of phishing URLs have obvious patterns")
        print("   → Requires sophisticated ML to distinguish")

    # Sample some phishing URLs
    print(f"\n{'='*80}")
    print("SAMPLE PHISHING URLs (first 10):")
    print(f"{'='*80}")
    for i, url in enumerate(phishing_urls[:10], 1):
        print(f"{i:2d}. {url[:80]}")

    # Analyze legitimate URLs
    print(f"\n{'='*80}")
    print("LEGITIMATE URL ANALYSIS")
    print(f"{'='*80}")

    legit_urls = legit['url'].tolist()

    # Extract domains
    common_domains = []
    for url in legit_urls[:20]:
        # Simple domain extraction
        domain = url.replace('http://', '').replace('https://', '').split('/')[0]
        common_domains.append(domain)

    print(f"\nSample legitimate domains (first 20):")
    domain_counts = Counter(common_domains)
    for domain, count in domain_counts.most_common(20):
        print(f"  - {domain}")

    # Check if legitimate URLs are all well-known sites
    well_known = ['google', 'facebook', 'amazon', 'microsoft', 'apple', 'twitter',
                  'linkedin', 'github', 'youtube', 'netflix', 'paypal', 'ebay']
    well_known_count = sum(1 for url in legit_urls
                           for site in well_known if site in url.lower())

    print(f"\nWell-known sites: {well_known_count} / {len(legit_urls)} ({well_known_count/len(legit_urls)*100:.1f}%)")

    if well_known_count / len(legit_urls) > 0.7:
        print("⚠️  Most legitimate URLs are well-known brands")
        print("   → Easy to distinguish from phishing")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_dataset_difficulty.py <dataset.csv>")
        print("\nExample: python analyze_dataset_difficulty.py dataset4.csv")
        sys.exit(1)

    analyze_dataset_difficulty(sys.argv[1])
