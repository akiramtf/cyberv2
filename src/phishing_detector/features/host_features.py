"""Host-based feature extraction"""

import socket
import ssl
from datetime import datetime
from typing import Dict, Optional
import dns.resolver
import dns.exception


class HostFeatures:
    """Extract host-based features from URLs"""

    def __init__(self, enable_dns: bool = True, enable_ssl: bool = True, enable_whois: bool = False, timeout: int = 5):
        self.enable_dns = enable_dns
        self.enable_ssl = enable_ssl
        self.enable_whois = enable_whois
        self.timeout = timeout
        self.dns_resolver = dns.resolver.Resolver()
        self.dns_resolver.timeout = timeout
        self.dns_resolver.lifetime = timeout

    def extract(self, url: str, components: Dict[str, any]) -> Dict[str, float]:
        """Extract all host-based features"""
        features = {}
        hostname = components.get("hostname", "")

        if not hostname:
            return self._get_default_features()

        # DNS-based features
        if self.enable_dns:
            dns_features = self._extract_dns_features(hostname)
            features.update(dns_features)
        else:
            features.update(self._get_default_dns_features())

        # SSL/TLS features
        if self.enable_ssl and components.get("scheme") == "https":
            port = components.get("port") or 443  # Handle None case
            ssl_features = self._extract_ssl_features(hostname, port)
            features.update(ssl_features)
        else:
            features.update(self._get_default_ssl_features())

        # Port features
        features["uses_standard_port"] = float(
            components.get("port") in [None, 80, 443]
        )
        features["uses_non_standard_port"] = float(
            components.get("port") not in [None, 80, 443]
        )

        return features

    def _extract_dns_features(self, hostname: str) -> Dict[str, float]:
        """Extract DNS-related features"""
        features = {}

        try:
            # DNS A record
            try:
                answers = self.dns_resolver.resolve(hostname, "A")
                features["has_dns_a_record"] = 1.0
                features["num_dns_a_records"] = float(len(answers))
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                features["has_dns_a_record"] = 0.0
                features["num_dns_a_records"] = 0.0

            # DNS MX record
            try:
                answers = self.dns_resolver.resolve(hostname, "MX")
                features["has_dns_mx_record"] = 1.0
                features["num_dns_mx_records"] = float(len(answers))
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                features["has_dns_mx_record"] = 0.0
                features["num_dns_mx_records"] = 0.0

            # DNS NS record
            try:
                answers = self.dns_resolver.resolve(hostname, "NS")
                features["has_dns_ns_record"] = 1.0
                features["num_dns_ns_records"] = float(len(answers))
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                features["has_dns_ns_record"] = 0.0
                features["num_dns_ns_records"] = 0.0

            # DNS TXT record
            try:
                answers = self.dns_resolver.resolve(hostname, "TXT")
                features["has_dns_txt_record"] = 1.0
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                features["has_dns_txt_record"] = 0.0

            # PTR record (reverse DNS)
            try:
                ip_address = socket.gethostbyname(hostname)
                reversed_ip = ".".join(reversed(ip_address.split("."))) + ".in-addr.arpa"
                answers = self.dns_resolver.resolve(reversed_ip, "PTR")
                features["has_ptr_record"] = 1.0
            except (
                socket.gaierror,
                dns.resolver.NXDOMAIN,
                dns.resolver.NoAnswer,
                dns.exception.Timeout,
            ):
                features["has_ptr_record"] = 0.0

        except Exception:
            features.update(self._get_default_dns_features())

        return features

    def _extract_ssl_features(self, hostname: str, port: int = 443) -> Dict[str, float]:
        """Extract SSL/TLS certificate features"""
        features = {}

        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()

                    # Certificate exists
                    features["has_ssl_cert"] = 1.0

                    # Certificate validity
                    if cert:
                        not_after = datetime.strptime(
                            cert["notAfter"], "%b %d %H:%M:%S %Y %Z"
                        )
                        not_before = datetime.strptime(
                            cert["notBefore"], "%b %d %H:%M:%S %Y %Z"
                        )
                        now = datetime.now()

                        features["ssl_cert_valid"] = float(
                            not_before <= now <= not_after
                        )

                        # Days until expiration
                        days_to_expire = (not_after - now).days
                        features["ssl_days_to_expire"] = float(max(0, days_to_expire))
                        features["ssl_cert_expires_soon"] = float(days_to_expire < 30)

                        # Certificate age
                        cert_age_days = (now - not_before).days
                        features["ssl_cert_age_days"] = float(max(0, cert_age_days))
                        features["ssl_cert_is_new"] = float(cert_age_days < 30)

                        # Subject Alternative Names
                        san = cert.get("subjectAltName", [])
                        features["ssl_num_san"] = float(len(san))
                    else:
                        features["ssl_cert_valid"] = 0.0
                        features["ssl_days_to_expire"] = 0.0
                        features["ssl_cert_expires_soon"] = 1.0
                        features["ssl_cert_age_days"] = 0.0
                        features["ssl_cert_is_new"] = 1.0
                        features["ssl_num_san"] = 0.0

        except (
            socket.gaierror,
            socket.timeout,
            ssl.SSLError,
            ConnectionRefusedError,
            OSError,
        ) as e:
            # Log the error for debugging
            import logging
            logging.debug(f"SSL extraction failed for {hostname}:{port} - {type(e).__name__}: {e}")
            features.update(self._get_default_ssl_features())

        return features

    def _get_default_features(self) -> Dict[str, float]:
        """Get default features when hostname is missing"""
        features = self._get_default_dns_features()
        features.update(self._get_default_ssl_features())
        features["uses_standard_port"] = 1.0
        features["uses_non_standard_port"] = 0.0
        return features

    def _get_default_dns_features(self) -> Dict[str, float]:
        """Default DNS features when DNS lookup is disabled or fails"""
        return {
            "has_dns_a_record": 0.0,
            "num_dns_a_records": 0.0,
            "has_dns_mx_record": 0.0,
            "num_dns_mx_records": 0.0,
            "has_dns_ns_record": 0.0,
            "num_dns_ns_records": 0.0,
            "has_dns_txt_record": 0.0,
            "has_ptr_record": 0.0,
        }

    def _get_default_ssl_features(self) -> Dict[str, float]:
        """Default SSL features when SSL check fails"""
        return {
            "has_ssl_cert": 0.0,
            "ssl_cert_valid": 0.0,
            "ssl_days_to_expire": 0.0,
            "ssl_cert_expires_soon": 1.0,
            "ssl_cert_age_days": 0.0,
            "ssl_cert_is_new": 1.0,
            "ssl_num_san": 0.0,
        }
