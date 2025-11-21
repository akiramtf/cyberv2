"""Create a balanced dataset from dataset5.csv"""

import pandas as pd
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def create_balanced_dataset(input_file="dataset5.csv", output_file="evaluation_test_data.csv", samples_per_class=1000):
    """Select equal number of samples for each label"""
    
    logger.info(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        logger.error(f"Error: {input_file} not found.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        sys.exit(1)

    # Ensure required columns exist
    if "label" not in df.columns:
        logger.error("Error: 'label' column not found in dataset.")
        sys.exit(1)

    # Separate by label
    legitimate = df[df["label"] == 0]
    phishing = df[df["label"] == 1]

    logger.info(f"Found {len(legitimate)} legitimate URLs and {len(phishing)} phishing URLs.")

    # Check if we have enough data
    if len(legitimate) < samples_per_class:
        logger.warning(f"Warning: Only {len(legitimate)} legitimate URLs available. Taking all of them.")
        samples_per_class_legit = len(legitimate)
    else:
        samples_per_class_legit = samples_per_class

    if len(phishing) < samples_per_class:
        logger.warning(f"Warning: Only {len(phishing)} phishing URLs available. Taking all of them.")
        samples_per_class_phishing = len(phishing)
    else:
        samples_per_class_phishing = samples_per_class

    # Sample data
    logger.info(f"Sampling {samples_per_class_legit} legitimate and {samples_per_class_phishing} phishing URLs...")
    
    legitimate_sample = legitimate.sample(n=samples_per_class_legit, random_state=42)
    phishing_sample = phishing.sample(n=samples_per_class_phishing, random_state=42)

    # Combine and shuffle
    balanced_df = pd.concat([legitimate_sample, phishing_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to file
    logger.info(f"Saving {len(balanced_df)} rows to {output_file}...")
    balanced_df.to_csv(output_file, index=False)
    logger.info("Done!")

if __name__ == "__main__":
    create_balanced_dataset()
