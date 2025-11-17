"""Ablation study: Test impact of different feature groups
Note: This script trains new models for each feature configuration.
For quick testing with pre-trained models, use test_individual_models.py instead.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.phishing_detector.detector import PhishingDetector
from src.phishing_detector.features.extractor import FeatureExtractor

def get_feature_indices(feature_names, feature_prefix):
    """Get indices of features matching a prefix"""
    return [i for i, name in enumerate(feature_names) if name.startswith(feature_prefix)]

def main():
    print("Loading dataset...")
    df = pd.read_csv('dataset2.csv', on_bad_lines='skip').head(5000)  # 5k for faster testing

    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor(enable_dns_lookup=False)

    features_list = []
    labels = []

    for _, row in df.iterrows():
        try:
            features = feature_extractor.extract_features(row['url'])
            features_list.append(features)
            labels.append(int(row['label']))
        except:
            pass

    df_features = pd.DataFrame(features_list)
    feature_names = df_features.columns.tolist()
    X_full = df_features.values
    y = np.array(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n" + "="*70)
    print("ABLATION STUDY: Feature Group Impact")
    print("="*70)

    # Baseline: All features (57 features)
    print("\n1. BASELINE - All 57 Features:")
    detector_full = PhishingDetector(enable_dns_lookup=False)
    detector_full.train(X_train, y_train)
    y_pred_full = (detector_full.ensemble_classifier.predict_proba(X_test) >= 0.5).astype(int)

    print(f"   Features: 57")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_full):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred_full):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_full):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_full):.4f}")

    # Lexical features only (40 features)
    print("\n2. Lexical Features Only:")
    lexical_features = [
        'url_length', 'hostname_length', 'path_length', 'query_length',
        'domain_length', 'subdomain_length', 'tld_length',
        'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
        'num_question_marks', 'num_equal_signs', 'num_at_symbols',
        'num_ampersands', 'num_percent_signs',
        'url_entropy', 'hostname_entropy', 'path_entropy',
        'digit_ratio', 'letter_ratio', 'digit_letter_ratio',
        'uppercase_ratio', 'lowercase_ratio',
        'path_depth', 'num_subdomains', 'num_query_params',
        'max_consecutive_digits', 'max_consecutive_letters',
        'has_ip_address', 'has_port', 'has_https',
        'has_double_slash_in_path', 'has_at_symbol', 'has_hyphen_in_domain',
        'is_url_shortener', 'has_suspicious_tld', 'has_query_params',
        'has_suspicious_keyword', 'num_suspicious_keywords'
    ]

    lexical_idx = [i for i, name in enumerate(feature_names) if name in lexical_features]
    X_train_lex = X_train[:, lexical_idx]
    X_test_lex = X_test[:, lexical_idx]

    detector_lex = PhishingDetector(enable_dns_lookup=False)
    detector_lex.train(X_train_lex, y_train)
    y_pred_lex = (detector_lex.ensemble_classifier.predict_proba(X_test_lex) >= 0.5).astype(int)

    acc_loss_lex = accuracy_score(y_test, y_pred_full) - accuracy_score(y_test, y_pred_lex)
    print(f"   Features: {len(lexical_idx)}")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_lex):.4f} (Δ{acc_loss_lex:+.4f})")
    print(f"   Precision: {precision_score(y_test, y_pred_lex):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_lex):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_lex):.4f}")

    # Without SSL features
    print("\n3. Without SSL/Certificate Features:")
    ssl_features = ['has_ssl_cert', 'ssl_cert_valid', 'ssl_days_to_expire',
                   'ssl_cert_expires_soon', 'ssl_cert_age_days', 'ssl_cert_is_new', 'ssl_num_san']

    no_ssl_idx = [i for i, name in enumerate(feature_names) if name not in ssl_features]
    X_train_no_ssl = X_train[:, no_ssl_idx]
    X_test_no_ssl = X_test[:, no_ssl_idx]

    detector_no_ssl = PhishingDetector(enable_dns_lookup=False)
    detector_no_ssl.train(X_train_no_ssl, y_train)
    y_pred_no_ssl = (detector_no_ssl.ensemble_classifier.predict_proba(X_test_no_ssl) >= 0.5).astype(int)

    acc_loss_ssl = accuracy_score(y_test, y_pred_full) - accuracy_score(y_test, y_pred_no_ssl)
    print(f"   Features: {len(no_ssl_idx)} (removed {len(ssl_features)} SSL features)")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_no_ssl):.4f} (Δ{acc_loss_ssl:+.4f})")
    print(f"   Precision: {precision_score(y_test, y_pred_no_ssl):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_no_ssl):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_no_ssl):.4f}")

    # Without DNS features
    print("\n4. Without DNS Features:")
    dns_features = ['has_dns_a_record', 'num_dns_a_records', 'has_dns_mx_record',
                   'num_dns_mx_records', 'has_dns_ns_record', 'num_dns_ns_records',
                   'has_dns_txt_record', 'has_ptr_record']

    no_dns_idx = [i for i, name in enumerate(feature_names) if name not in dns_features]
    X_train_no_dns = X_train[:, no_dns_idx]
    X_test_no_dns = X_test[:, no_dns_idx]

    detector_no_dns = PhishingDetector(enable_dns_lookup=False)
    detector_no_dns.train(X_train_no_dns, y_train)
    y_pred_no_dns = (detector_no_dns.ensemble_classifier.predict_proba(X_test_no_dns) >= 0.5).astype(int)

    acc_loss_dns = accuracy_score(y_test, y_pred_full) - accuracy_score(y_test, y_pred_no_dns)
    print(f"   Features: {len(no_dns_idx)} (removed {len(dns_features)} DNS features)")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred_no_dns):.4f} (Δ{acc_loss_dns:+.4f})")
    print(f"   Precision: {precision_score(y_test, y_pred_no_dns):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred_no_dns):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred_no_dns):.4f}")

    print("\n" + "="*70)
    print("FEATURE GROUP CONTRIBUTION RANKING")
    print("="*70)
    contributions = {
        "SSL Features": acc_loss_ssl,
        "DNS Features": acc_loss_dns,
    }

    for name, loss in sorted(contributions.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {loss:.4f} accuracy loss when removed")

if __name__ == "__main__":
    main()
