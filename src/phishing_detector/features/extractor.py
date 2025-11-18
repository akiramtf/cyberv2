"""Main feature extractor that combines all feature types"""

from typing import Dict, List
import pandas as pd

from ..utils.url_parser import URLParser
from .lexical import LexicalFeatures
from .host_features import HostFeatures


class FeatureExtractor:
    """Extract all features from URLs"""

    def __init__(
        self,
        enable_dns_lookup: bool = True,
        enable_whois_lookup: bool = False,
        enable_ssl_check: bool = True,
        timeout: int = 5,
    ):
        self.url_parser = URLParser()
        self.lexical_extractor = LexicalFeatures()
        self.host_extractor = HostFeatures(
            enable_dns=enable_dns_lookup,
            enable_whois=enable_whois_lookup,
            enable_ssl=enable_ssl_check,
            timeout=timeout,
        )

    def extract_features(self, url: str) -> Dict[str, float]:
        """Extract all features from a single URL"""
        try:
            # Parse URL
            components = self.url_parser.parse(url)

            # Extract features
            features = {}

            # Lexical features (40+ features)
            lexical_features = self.lexical_extractor.extract(url, components)
            features.update(lexical_features)

            # Host features (15+ features)
            host_features = self.host_extractor.extract(url, components)
            features.update(host_features)

            return features

        except Exception as e:
            # Return default features on error
            return self._get_default_features()

    def extract_batch(self, urls: List[str]) -> pd.DataFrame:
        """Extract features from multiple URLs"""
        features_list = []

        for url in urls:
            features = self.extract_features(url)
            features["url"] = url
            features_list.append(features)

        df = pd.DataFrame(features_list)

        # Move URL column to first position
        cols = ["url"] + [col for col in df.columns if col != "url"]
        df = df[cols]

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # Generate feature names from a sample URL
        sample_features = self.extract_features("https://example.com")
        return sorted(sample_features.keys())

    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values when extraction fails"""
        # Default lexical features
        features = {
            "url_length": 0.0,
            "hostname_length": 0.0,
            "path_length": 0.0,
            "query_length": 0.0,
            "path_depth": 0.0,
            "num_dots": 0.0,
            "num_hyphens": 0.0,
            "num_underscores": 0.0,
            "num_slashes": 0.0,
            "num_question_marks": 0.0,
            "num_equal_signs": 0.0,
            "num_at_symbols": 0.0,
            "num_ampersands": 0.0,
            "num_percent_signs": 0.0,
            "has_ip_address": 0.0,
            "has_port": 0.0,
            "has_https": 0.0,
            "has_double_slash_in_path": 0.0,
            "has_at_symbol": 0.0,
            "has_hyphen_in_domain": 0.0,
            "url_entropy": 0.0,
            "hostname_entropy": 0.0,
            "path_entropy": 0.0,
            "digit_ratio": 0.0,
            "letter_ratio": 0.0,
            "digit_letter_ratio": 0.0,
            "domain_length": 0.0,
            "subdomain_length": 0.0,
            "tld_length": 0.0,
            "num_subdomains": 0.0,
            "is_url_shortener": 0.0,
            "has_suspicious_tld": 0.0,
            "num_query_params": 0.0,
            "has_query_params": 0.0,
            "uppercase_ratio": 0.0,
            "lowercase_ratio": 0.0,
            "max_consecutive_digits": 0.0,
            "max_consecutive_letters": 0.0,
            "has_suspicious_keyword": 0.0,
            "num_suspicious_keywords": 0.0,
        }

        # Default host features
        features.update(
            {
                "has_dns_a_record": 0.0,
                "num_dns_a_records": 0.0,
                "has_dns_mx_record": 0.0,
                "num_dns_mx_records": 0.0,
                "has_dns_ns_record": 0.0,
                "num_dns_ns_records": 0.0,
                "has_dns_txt_record": 0.0,
                "has_ptr_record": 0.0,
                "has_ssl_cert": 0.0,
                "ssl_cert_valid": 0.0,
                "ssl_days_to_expire": 0.0,
                "ssl_cert_expires_soon": 1.0,
                "ssl_cert_age_days": 0.0,
                "ssl_cert_is_new": 1.0,
                "ssl_num_san": 0.0,
                "uses_standard_port": 1.0,
                "uses_non_standard_port": 0.0,
            }
        )

        return features
