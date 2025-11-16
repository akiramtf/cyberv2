"""Prepare training data from PhishTank CSV + legitimate URLs"""

import pandas as pd
import sys
import os

# Add training directory to path to import legitimate URLs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

try:
    from train_improved import load_improved_sample_data
    USE_IMPROVED_DATA = True
except ImportError:
    USE_IMPROVED_DATA = False

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

    # Legitimate URLs will be loaded from train_improved.py (120+ URLs)
    # This provides both www/non-www variants and legitimate subdomains
    legitimate_urls = []

    # Load legitimate URLs from train_improved.py
    if USE_IMPROVED_DATA:
        print(f"Loading legitimate URLs from training data...")
        try:
            df_improved = load_improved_sample_data()
            legitimate_urls = df_improved[df_improved['label'] == 0]['url'].tolist()
            print(f"Loaded {len(legitimate_urls)} legitimate URLs")
        except Exception as e:
            print(f"⚠️  Error: Could not load legitimate URLs: {e}")
            print(f"   Please ensure train_improved.py is available.")
            sys.exit(1)
    else:
        print(f"⚠️  Error: Could not import train_improved.py")
        print(f"   Please ensure the training module is available.")
        sys.exit(1)

    # Match the number of legitimate URLs to phishing URLs
    num_legitimate_needed = min(len(phishing_urls), len(legitimate_urls))

    if num_legitimate_needed < len(phishing_urls):
        print(f"⚠️  WARNING: Only {num_legitimate_needed} legitimate URLs available,")
        print(f"   but you requested {len(phishing_urls)} phishing samples.")
        print(f"   Using {num_legitimate_needed} of each for balanced training.")

    legitimate_urls = legitimate_urls[:num_legitimate_needed]
    phishing_urls = phishing_urls[:num_legitimate_needed]

    print(f"Using {len(legitimate_urls)} legitimate URLs")
    print(f"Using {len(phishing_urls)} phishing URLs")

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
