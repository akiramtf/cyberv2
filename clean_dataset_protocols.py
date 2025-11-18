#!/usr/bin/env python3
"""
Clean Dataset - Remove HTTP/HTTPS Protocols
This removes protocols from URLs to avoid slow SSL certificate checks during training.
"""

import sys
import pandas as pd
import argparse


def clean_dataset(input_file, output_file=None):
    """Remove http:// and https:// from URLs"""

    print(f"Loading dataset: {input_file}")
    df = pd.read_csv(input_file, on_bad_lines='skip')

    print(f"Total rows: {len(df):,}")

    if 'url' not in df.columns:
        print("❌ Error: No 'url' column found!")
        return

    # Count protocols before cleaning
    https_count = df['url'].str.startswith('https://').sum()
    http_count = df['url'].str.startswith('http://').sum()

    print(f"\nBefore cleaning:")
    print(f"  HTTPS URLs: {https_count:,}")
    print(f"  HTTP URLs:  {http_count:,}")
    print(f"  Others:     {len(df) - https_count - http_count:,}")

    # Remove protocols
    df['url'] = df['url'].str.replace(r'^https?://', '', regex=True)

    # Verify cleaning
    https_after = df['url'].str.startswith('https://').sum()
    http_after = df['url'].str.startswith('http://').sum()

    print(f"\nAfter cleaning:")
    print(f"  URLs with protocol: {https_after + http_after:,}")
    print(f"  URLs without protocol: {len(df) - https_after - http_after:,}")

    # Save
    if output_file is None:
        output_file = input_file.replace('.csv', '_cleaned.csv')

    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")

    # Time estimate
    time_before = (https_count * 5 + (len(df) - https_count) * 0.05) / 60  # minutes
    time_after = len(df) * 0.05 / 60  # minutes

    print(f"\n⏱️  Training time estimate:")
    print(f"  Before: ~{time_before:.1f} minutes ({time_before/60:.1f} hours)")
    print(f"  After:  ~{time_after:.1f} minutes")
    print(f"  Speedup: {time_before/time_after:.1f}x faster!")


def main():
    parser = argparse.ArgumentParser(description="Remove HTTP/HTTPS protocols from dataset URLs")
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('-o', '--output', help='Output CSV file (default: input_cleaned.csv)')

    args = parser.parse_args()

    try:
        clean_dataset(args.input, args.output)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
