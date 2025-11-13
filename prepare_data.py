"""Prepare training data from PhishTank CSV + legitimate URLs"""

import pandas as pd
import sys

def prepare_training_data(phishing_csv_path, output_path="data/processed/training_data.csv", max_samples=1000):
    """
    Prepare training data combining phishing URLs with legitimate URLs

    Args:
        phishing_csv_path: Path to PhishTank CSV file
        output_path: Where to save the prepared data
        max_samples: Maximum phishing samples to use (will match with equal legitimate URLs)
    """

    print(f"Loading phishing data from: {phishing_csv_path}")

    # Read the phishing CSV (only need the 'url' column)
    try:
        df_phishing = pd.read_csv(phishing_csv_path)
        print(f"Loaded {len(df_phishing)} rows")
        print(f"Columns found: {list(df_phishing.columns)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Extract only the URL column
    if 'url' not in df_phishing.columns:
        print("ERROR: 'url' column not found!")
        print(f"Available columns: {list(df_phishing.columns)}")
        sys.exit(1)

    # Get phishing URLs and limit to max_samples
    phishing_urls = df_phishing['url'].dropna().unique()[:max_samples]
    print(f"\nUsing {len(phishing_urls)} phishing URLs")

    # Legitimate URLs (popular, trusted sites)
    legitimate_urls = [
        # Search engines
        "https://www.google.com",
        "https://www.bing.com",
        "https://www.yahoo.com",
        "https://duckduckgo.com",
        "https://www.baidu.com",
        # Social media
        "https://www.facebook.com",
        "https://www.twitter.com",
        "https://www.instagram.com",
        "https://www.linkedin.com",
        "https://www.reddit.com",
        "https://www.pinterest.com",
        "https://www.tumblr.com",
        "https://www.snapchat.com",
        "https://www.tiktok.com",
        # Tech companies
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.amazon.com",
        "https://www.google.com/gmail",
        "https://www.icloud.com",
        "https://www.samsung.com",
        "https://www.sony.com",
        "https://www.adobe.com",
        "https://www.oracle.com",
        "https://www.ibm.com",
        "https://www.intel.com",
        "https://www.nvidia.com",
        # E-commerce
        "https://www.ebay.com",
        "https://www.etsy.com",
        "https://www.walmart.com",
        "https://www.target.com",
        "https://www.bestbuy.com",
        "https://www.aliexpress.com",
        "https://www.alibaba.com",
        # Streaming
        "https://www.netflix.com",
        "https://www.youtube.com",
        "https://www.spotify.com",
        "https://www.hulu.com",
        "https://www.twitch.tv",
        "https://www.disneyplus.com",
        "https://www.primevideo.com",
        # Development
        "https://www.github.com",
        "https://stackoverflow.com",
        "https://www.gitlab.com",
        "https://www.bitbucket.org",
        "https://www.docker.com",
        "https://www.python.org",
        "https://www.java.com",
        "https://www.ruby-lang.org",
        "https://www.php.net",
        "https://www.golang.org",
        # Cloud
        "https://aws.amazon.com",
        "https://cloud.google.com",
        "https://azure.microsoft.com",
        "https://www.digitalocean.com",
        "https://www.heroku.com",
        # News
        "https://www.cnn.com",
        "https://www.bbc.com",
        "https://www.nytimes.com",
        "https://www.theguardian.com",
        "https://www.reuters.com",
        "https://www.bloomberg.com",
        "https://www.wsj.com",
        "https://www.forbes.com",
        # Education
        "https://www.wikipedia.org",
        "https://www.coursera.org",
        "https://www.udemy.com",
        "https://www.edx.org",
        "https://www.khanacademy.org",
        "https://www.mit.edu",
        "https://www.stanford.edu",
        "https://www.harvard.edu",
        # Finance
        "https://www.paypal.com",
        "https://www.stripe.com",
        "https://www.square.com",
        "https://www.venmo.com",
        "https://www.chase.com",
        "https://www.bankofamerica.com",
        "https://www.wellsfargo.com",
        "https://www.citibank.com",
        # Productivity
        "https://www.dropbox.com",
        "https://drive.google.com",
        "https://onedrive.live.com",
        "https://www.zoom.us",
        "https://www.slack.com",
        "https://www.trello.com",
        "https://www.notion.so",
        "https://www.asana.com",
        "https://www.monday.com",
        # Email
        "https://mail.google.com",
        "https://outlook.live.com",
        "https://mail.yahoo.com",
        "https://www.protonmail.com",
        # Travel
        "https://www.booking.com",
        "https://www.airbnb.com",
        "https://www.expedia.com",
        "https://www.tripadvisor.com",
        # Other popular sites
        "https://www.wordpress.com",
        "https://www.blogger.com",
        "https://www.medium.com",
        "https://www.quora.com",
        "https://www.imdb.com",
        "https://www.yelp.com",
    ]

    # Match the number of legitimate URLs to phishing URLs
    num_legitimate_needed = min(len(phishing_urls), len(legitimate_urls))
    legitimate_urls = legitimate_urls[:num_legitimate_needed]
    phishing_urls = phishing_urls[:num_legitimate_needed]

    print(f"Using {len(legitimate_urls)} legitimate URLs")

    # Create combined DataFrame
    df_combined = pd.DataFrame({
        'url': list(legitimate_urls) + list(phishing_urls),
        'label': [0] * len(legitimate_urls) + [1] * len(phishing_urls)
    })

    # Shuffle the data
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False)

    print(f"\n✅ Training data prepared successfully!")
    print(f"   Saved to: {output_path}")
    print(f"   Total URLs: {len(df_combined)}")
    print(f"   Legitimate: {(df_combined['label'] == 0).sum()}")
    print(f"   Phishing: {(df_combined['label'] == 1).sum()}")
    print(f"\nNow run: python training/train_improved.py --data {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data from PhishTank CSV")
    parser.add_argument("phishing_csv", help="Path to PhishTank CSV file")
    parser.add_argument("--output", default="data/processed/training_data.csv",
                       help="Output path for prepared data")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Maximum number of phishing samples to use")

    args = parser.parse_args()

    prepare_training_data(args.phishing_csv, args.output, args.max_samples)
