#!/usr/bin/env python3
"""
Dataset Diagnostics Tool
Analyzes datasets to predict training time and identify issues.
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path


def analyze_dataset(file_path):
    """Analyze a dataset file and estimate training time"""

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return

    print(f"\n{'='*80}")
    print(f"ANALYZING: {file_path}")
    print(f"{'='*80}\n")

    # File size
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📁 File Size: {file_size_mb:.2f} MB ({file_size:,} bytes)")

    # Load dataset
    try:
        print(f"📖 Loading dataset...")
        df = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"✅ Successfully loaded")
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return

    # Basic stats
    total_rows = len(df)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Columns: {list(df.columns)}")

    # Check for required columns
    if 'url' not in df.columns or 'label' not in df.columns:
        print(f"❌ Missing required columns! Need 'url' and 'label'")
        return

    # Clean data
    df_clean = df.dropna(subset=['url', 'label'])
    df_clean = df_clean[df_clean['url'].str.strip() != '']
    valid_rows = len(df_clean)
    print(f"   Valid rows (after cleaning): {valid_rows:,}")
    print(f"   Invalid rows: {total_rows - valid_rows:,}")

    # Label distribution
    if 'label' in df_clean.columns:
        try:
            df_clean['label'] = pd.to_numeric(df_clean['label'], errors='coerce')
            df_clean = df_clean.dropna(subset=['label'])

            legitimate = len(df_clean[df_clean['label'] == 0])
            phishing = len(df_clean[df_clean['label'] == 1])

            print(f"\n🏷️  Label Distribution:")
            print(f"   Legitimate (0): {legitimate:,} ({legitimate/valid_rows*100:.2f}%)")
            print(f"   Phishing (1):   {phishing:,} ({phishing/valid_rows*100:.2f}%)")

            # Check imbalance
            if legitimate > 0 and phishing > 0:
                ratio = max(legitimate, phishing) / min(legitimate, phishing)
                print(f"   Imbalance Ratio: {ratio:.1f}:1")

                if ratio > 10:
                    print(f"\n   ⚠️  WARNING: Severe class imbalance!")
                    if legitimate < 1000 or phishing < 1000:
                        print(f"   💡 Recommendation: Need at least 1,000 samples per class")
        except Exception as e:
            print(f"   ⚠️  Could not analyze labels: {e}")

    # Sample URLs
    print(f"\n🔗 Sample URLs:")
    for i, url in enumerate(df_clean['url'].head(5), 1):
        print(f"   {i}. {url[:80]}...")

    # Estimate training time
    print(f"\n⏱️  TRAINING TIME ESTIMATES:")
    print(f"{'='*80}")

    # Time per URL for feature extraction
    time_per_url_no_dns = 0.05  # 50ms without DNS
    time_per_url_with_dns = 15.0  # 15 seconds with DNS (average)

    # Estimate without DNS
    total_time_no_dns = valid_rows * time_per_url_no_dns / 60  # minutes
    print(f"\n✅ WITHOUT DNS LOOKUPS (enable_dns_lookup=False):")
    if total_time_no_dns < 1:
        print(f"   Feature Extraction: ~{total_time_no_dns * 60:.0f} seconds")
    elif total_time_no_dns < 60:
        print(f"   Feature Extraction: ~{total_time_no_dns:.1f} minutes")
    else:
        print(f"   Feature Extraction: ~{total_time_no_dns / 60:.1f} hours")

    # Model training time (rough estimate)
    model_training_minutes = (valid_rows / 1000) * 0.5  # ~30 seconds per 1000 samples
    total_no_dns = total_time_no_dns + model_training_minutes

    if total_no_dns < 1:
        print(f"   Model Training: ~{model_training_minutes * 60:.0f} seconds")
        print(f"   TOTAL: ~{total_no_dns * 60:.0f} seconds")
    elif total_no_dns < 60:
        print(f"   Model Training: ~{model_training_minutes:.1f} minutes")
        print(f"   TOTAL: ~{total_no_dns:.1f} minutes")
    else:
        print(f"   Model Training: ~{model_training_minutes / 60:.1f} hours")
        print(f"   TOTAL: ~{total_no_dns / 60:.1f} hours")

    # Estimate with DNS
    total_time_with_dns = valid_rows * time_per_url_with_dns / 60  # minutes
    print(f"\n❌ WITH DNS LOOKUPS (enable_dns_lookup=True): ⚠️  SLOW!")
    if total_time_with_dns < 60:
        print(f"   Feature Extraction: ~{total_time_with_dns:.1f} minutes")
    else:
        print(f"   Feature Extraction: ~{total_time_with_dns / 60:.1f} hours")

    total_with_dns = total_time_with_dns + model_training_minutes
    if total_with_dns < 60:
        print(f"   TOTAL: ~{total_with_dns:.1f} minutes")
    else:
        print(f"   TOTAL: ~{total_with_dns / 60:.1f} hours ({total_with_dns / 60 / 24:.1f} days)")

    print(f"\n{'='*80}")
    print(f"💡 RECOMMENDATIONS:")
    print(f"{'='*80}")

    # Recommendations based on size
    if valid_rows > 50000:
        print(f"📌 Large dataset ({valid_rows:,} rows)")
        print(f"   → Use --max-rows to limit training size for faster experiments")
        print(f"   → Example: python training/train.py --data {file_path} --max-rows 10000")

    # DNS recommendation
    print(f"\n📌 ALWAYS disable DNS lookups for training:")
    print(f"   → Check training/train.py line 707: enable_dns_lookup=False")
    print(f"   → DNS lookups make training {int(time_per_url_with_dns / time_per_url_no_dns)}x slower!")

    # Imbalance recommendation
    if legitimate > 0 and phishing > 0:
        ratio = max(legitimate, phishing) / min(legitimate, phishing)
        if ratio > 10:
            print(f"\n📌 Fix class imbalance (ratio: {ratio:.1f}:1):")
            print(f"   → Add more samples of minority class")
            print(f"   → Or use class_weight='balanced' in models")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset and estimate training time")
    parser.add_argument('files', nargs='+', help='Dataset CSV files to analyze')

    args = parser.parse_args()

    for file_path in args.files:
        analyze_dataset(file_path)

    # Compare if multiple files
    if len(args.files) > 1:
        print(f"\n{'='*80}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*80}\n")

        for file_path in args.files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, on_bad_lines='skip')
                    df_clean = df.dropna(subset=['url', 'label'])
                    rows = len(df_clean)
                    time_est = rows * 0.05 / 60  # minutes without DNS

                    print(f"{os.path.basename(file_path):20} → {rows:8,} rows → ~{time_est:6.1f} min (no DNS)")
                except:
                    print(f"{os.path.basename(file_path):20} → Error loading")

        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
