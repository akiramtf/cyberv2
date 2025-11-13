"""Lexical feature extraction from URLs"""

import math
import re
from collections import Counter
from typing import Dict
from urllib.parse import urlparse


class LexicalFeatures:
    """Extract lexical features from URLs"""

    def __init__(self):
        self.suspicious_chars = ["@", "//", "-", "_"]
        self.shortening_services = [
            "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "is.gd", "buff.ly",
            "adf.ly", "t.co", "lnkd.in", "tiny.cc"
        ]

    def extract(self, url: str, components: Dict[str, any]) -> Dict[str, float]:
        """Extract all lexical features"""
        features = {}

        # Basic length features
        features["url_length"] = len(url)
        features["hostname_length"] = len(components["hostname"])
        features["path_length"] = len(components["path"])
        features["query_length"] = len(components["query"])

        # URL depth (number of subdirectories)
        features["path_depth"] = components["path"].count("/")

        # Count special characters
        features["num_dots"] = url.count(".")
        features["num_hyphens"] = url.count("-")
        features["num_underscores"] = url.count("_")
        features["num_slashes"] = url.count("/")
        features["num_question_marks"] = url.count("?")
        features["num_equal_signs"] = url.count("=")
        features["num_at_symbols"] = url.count("@")
        features["num_ampersands"] = url.count("&")
        features["num_percent_signs"] = url.count("%")

        # Binary features
        features["has_ip_address"] = float(self._is_ip_address(components["hostname"]))
        features["has_port"] = float(components["port"] is not None)
        features["has_https"] = float(components["scheme"] == "https")
        features["has_www"] = float("www" in components["subdomain"].lower())

        # Suspicious patterns
        features["has_double_slash_in_path"] = float("//" in components["path"])
        features["has_at_symbol"] = float("@" in url)
        features["has_hyphen_in_domain"] = float("-" in components["domain"])

        # Entropy features
        features["url_entropy"] = self._calculate_entropy(url)
        features["hostname_entropy"] = self._calculate_entropy(components["hostname"])
        features["path_entropy"] = self._calculate_entropy(components["path"])

        # Digit and letter ratios
        features["digit_ratio"] = self._digit_ratio(url)
        features["letter_ratio"] = self._letter_ratio(url)
        features["digit_letter_ratio"] = (
            features["digit_ratio"] / features["letter_ratio"]
            if features["letter_ratio"] > 0 else 0
        )

        # Domain features
        features["domain_length"] = len(components["domain"])
        features["subdomain_length"] = len(components["subdomain"])
        features["tld_length"] = len(components["suffix"])
        features["num_subdomains"] = (
            len(components["subdomain"].split("."))
            if components["subdomain"] else 0
        )

        # Suspicious patterns
        features["is_url_shortener"] = float(
            any(service in components["hostname"] for service in self.shortening_services)
        )
        features["has_suspicious_tld"] = float(
            components["suffix"].lower() in {
                "tk", "ml", "ga", "cf", "gq", "work", "click", "link", "top"
            }
        )

        # Query parameter features
        num_params = len(components.get("query_params", {}))
        features["num_query_params"] = num_params
        features["has_query_params"] = float(num_params > 0)

        # Uppercase/lowercase ratio
        features["uppercase_ratio"] = self._uppercase_ratio(url)
        features["lowercase_ratio"] = self._lowercase_ratio(url)

        # Number of consecutive characters
        features["max_consecutive_digits"] = self._max_consecutive(url, str.isdigit)
        features["max_consecutive_letters"] = self._max_consecutive(url, str.isalpha)

        # Suspicious keywords
        suspicious_keywords = [
            "login", "signin", "account", "update", "verify", "secure", "banking",
            "paypal", "ebay", "amazon", "apple", "microsoft", "confirm", "suspended"
        ]
        url_lower = url.lower()
        features["has_suspicious_keyword"] = float(
            any(keyword in url_lower for keyword in suspicious_keywords)
        )
        features["num_suspicious_keywords"] = sum(
            1 for keyword in suspicious_keywords if keyword in url_lower
        )

        return features

    def _calculate_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not string:
            return 0.0

        counter = Counter(string)
        length = len(string)
        entropy = 0.0

        for count in counter.values():
            probability = count / length
            entropy -= probability * math.log2(probability)

        return entropy

    def _digit_ratio(self, string: str) -> float:
        """Calculate ratio of digits in string"""
        if not string:
            return 0.0
        digits = sum(1 for char in string if char.isdigit())
        return digits / len(string)

    def _letter_ratio(self, string: str) -> float:
        """Calculate ratio of letters in string"""
        if not string:
            return 0.0
        letters = sum(1 for char in string if char.isalpha())
        return letters / len(string)

    def _uppercase_ratio(self, string: str) -> float:
        """Calculate ratio of uppercase letters"""
        if not string:
            return 0.0
        uppercase = sum(1 for char in string if char.isupper())
        return uppercase / len(string)

    def _lowercase_ratio(self, string: str) -> float:
        """Calculate ratio of lowercase letters"""
        if not string:
            return 0.0
        lowercase = sum(1 for char in string if char.islower())
        return lowercase / len(string)

    def _max_consecutive(self, string: str, check_func) -> int:
        """Find maximum consecutive characters matching a condition"""
        if not string:
            return 0

        max_count = 0
        current_count = 0

        for char in string:
            if check_func(char):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def _is_ip_address(self, hostname: str) -> bool:
        """Check if hostname is an IP address"""
        if not hostname:
            return False
        ip_pattern = re.compile(
            r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        )
        return bool(ip_pattern.match(hostname))
