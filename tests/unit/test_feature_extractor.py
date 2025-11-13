"""Unit tests for feature extraction"""

import pytest
from src.phishing_detector.features.extractor import FeatureExtractor
from src.phishing_detector.features.lexical import LexicalFeatures
from src.phishing_detector.utils.url_parser import URLParser


class TestURLParser:
    """Test URL parser"""

    def test_parse_valid_url(self):
        """Test parsing a valid URL"""
        parser = URLParser()
        result = parser.parse("https://www.example.com/path?query=1")

        assert result["scheme"] == "https"
        assert result["domain"] == "example"
        assert result["subdomain"] == "www"
        assert result["suffix"] == "com"
        assert result["path"] == "/path"
        assert result["query"] == "query=1"

    def test_parse_url_without_scheme(self):
        """Test parsing URL without scheme"""
        parser = URLParser()
        result = parser.parse("example.com")

        assert result["domain"] == "example"
        assert result["suffix"] == "com"

    def test_is_ip_address(self):
        """Test IP address detection"""
        parser = URLParser()

        assert parser.is_ip_address("192.168.1.1") is True
        assert parser.is_ip_address("example.com") is False

    def test_has_suspicious_tld(self):
        """Test suspicious TLD detection"""
        parser = URLParser()

        assert parser.has_suspicious_tld("tk") is True
        assert parser.has_suspicious_tld("ml") is True
        assert parser.has_suspicious_tld("com") is False

    def test_has_suspicious_keywords(self):
        """Test suspicious keyword detection"""
        parser = URLParser()

        assert parser.has_suspicious_keywords("https://paypal-login.com") is True
        assert parser.has_suspicious_keywords("https://example.com") is False


class TestLexicalFeatures:
    """Test lexical feature extraction"""

    def test_extract_basic_features(self):
        """Test basic feature extraction"""
        extractor = LexicalFeatures()
        parser = URLParser()

        url = "https://www.example.com/path"
        components = parser.parse(url)
        features = extractor.extract(url, components)

        assert "url_length" in features
        assert "hostname_length" in features
        assert "path_length" in features
        assert features["url_length"] > 0

    def test_entropy_calculation(self):
        """Test entropy calculation"""
        extractor = LexicalFeatures()

        # Low entropy
        entropy1 = extractor._calculate_entropy("aaaaaaa")

        # High entropy
        entropy2 = extractor._calculate_entropy("abcdefg")

        assert entropy2 > entropy1

    def test_digit_ratio(self):
        """Test digit ratio calculation"""
        extractor = LexicalFeatures()

        assert extractor._digit_ratio("123abc") == 0.5
        assert extractor._digit_ratio("abc") == 0.0
        assert extractor._digit_ratio("123") == 1.0

    def test_suspicious_patterns(self):
        """Test suspicious pattern detection"""
        extractor = LexicalFeatures()
        parser = URLParser()

        # Phishing-like URL
        url = "http://paypal-verify.suspicious.tk/login"
        components = parser.parse(url)
        features = extractor.extract(url, components)

        assert features["has_suspicious_keyword"] == 1.0
        assert features["has_suspicious_tld"] == 1.0

    def test_url_with_ip(self):
        """Test URL with IP address"""
        extractor = LexicalFeatures()
        parser = URLParser()

        url = "http://192.168.1.1/path"
        components = parser.parse(url)
        features = extractor.extract(url, components)

        assert features["has_ip_address"] == 1.0


class TestFeatureExtractor:
    """Test complete feature extraction"""

    def test_extract_features_legitimate(self):
        """Test feature extraction for legitimate URL"""
        extractor = FeatureExtractor(enable_dns_lookup=False)

        features = extractor.extract_features("https://www.google.com")

        assert isinstance(features, dict)
        assert len(features) > 40  # Should have 50+ features
        assert "url_length" in features
        assert "has_https" in features

    def test_extract_features_phishing(self):
        """Test feature extraction for phishing-like URL"""
        extractor = FeatureExtractor(enable_dns_lookup=False)

        features = extractor.extract_features("http://paypal-verify.tk/login")

        assert isinstance(features, dict)
        assert features["has_suspicious_keyword"] == 1.0

    def test_extract_batch(self):
        """Test batch feature extraction"""
        extractor = FeatureExtractor(enable_dns_lookup=False)

        urls = [
            "https://www.google.com",
            "https://www.example.com",
            "http://suspicious.tk",
        ]

        df = extractor.extract_batch(urls)

        assert len(df) == 3
        assert "url" in df.columns
        assert "url_length" in df.columns

    def test_get_feature_names(self):
        """Test getting feature names"""
        extractor = FeatureExtractor(enable_dns_lookup=False)

        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 40

    def test_invalid_url(self):
        """Test handling of invalid URL"""
        extractor = FeatureExtractor(enable_dns_lookup=False)

        # Should return default features without crashing
        features = extractor.extract_features("not-a-valid-url")

        assert isinstance(features, dict)
