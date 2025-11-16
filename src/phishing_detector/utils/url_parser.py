"""URL parsing utilities"""

import re
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import validators
import tldextract


class URLParser:
    """Parse and extract components from URLs"""

    def __init__(self):
        self.ip_pattern = re.compile(
            r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        )
        self.suspicious_tlds = {
            "tk", "ml", "ga", "cf", "gq", "work", "click", "link", "top", "xyz",
            "bid", "science", "win", "download", "racing", "accountant", "date"
        }
        self.suspicious_keywords = [
            "login", "signin", "account", "update", "verify", "secure", "banking",
            "paypal", "ebay", "amazon", "apple", "microsoft", "confirm", "suspended",
            "locked", "password", "credential", "verification"
        ]

    def parse(self, url: str) -> Dict[str, any]:
        """Parse URL and extract all components"""
        if not url:
            raise ValueError("URL cannot be empty")

        # Ensure URL has a scheme
        if not url.startswith(("http://", "https://", "ftp://")):
            url = "http://" + url

        # Validate URL
        if not validators.url(url):
            raise ValueError(f"Invalid URL: {url}")

        parsed = urlparse(url)
        extracted = tldextract.extract(url)

        # Normalize subdomain: remove "www." prefix to treat www.example.com same as example.com
        subdomain = extracted.subdomain or ""
        if subdomain.lower() == "www":
            subdomain = ""
        elif subdomain.lower().startswith("www."):
            subdomain = subdomain[4:]  # Remove "www." prefix

        components = {
            "url": url,
            "scheme": parsed.scheme or "",
            "netloc": parsed.netloc or "",
            "path": parsed.path or "",
            "params": parsed.params or "",
            "query": parsed.query or "",
            "fragment": parsed.fragment or "",
            "username": parsed.username or "",
            "password": parsed.password or "",
            "hostname": parsed.hostname or "",
            "port": parsed.port,
            "domain": extracted.domain or "",
            "subdomain": subdomain,  # Use normalized subdomain
            "suffix": extracted.suffix or "",
            "registered_domain": extracted.registered_domain or "",
            "fqdn": extracted.fqdn or "",
        }

        # Query parameters
        components["query_params"] = parse_qs(parsed.query) if parsed.query else {}

        return components

    def is_ip_address(self, hostname: str) -> bool:
        """Check if hostname is an IP address"""
        if not hostname:
            return False
        return bool(self.ip_pattern.match(hostname))

    def has_suspicious_tld(self, suffix: str) -> bool:
        """Check if TLD is suspicious"""
        return suffix.lower() in self.suspicious_tlds

    def has_suspicious_keywords(self, url: str) -> bool:
        """Check if URL contains suspicious keywords"""
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in self.suspicious_keywords)

    def count_subdomains(self, subdomain: str) -> int:
        """Count number of subdomains"""
        if not subdomain:
            return 0
        return len(subdomain.split("."))

    def extract_port(self, url: str) -> Optional[int]:
        """Extract port from URL"""
        parsed = urlparse(url)
        return parsed.port

    def is_https(self, url: str) -> bool:
        """Check if URL uses HTTPS"""
        return url.startswith("https://")
