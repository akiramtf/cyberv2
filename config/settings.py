"""Configuration settings for the phishing detector"""

import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    api_reload: bool = Field(default=False)

    # Model Configuration
    model_path: str = Field(default="models/trained")
    model_version: str = Field(default="v1")
    enable_zero_day_detection: bool = Field(default=False)  # Disabled by default - causes false positives with small datasets
    anomaly_threshold: float = Field(default=0.7)  # Higher threshold = less sensitive

    # Feature Extraction
    enable_dns_lookup: bool = Field(default=True)
    enable_whois_lookup: bool = Field(default=False)
    enable_content_fetch: bool = Field(default=False)
    request_timeout: int = Field(default=5)

    # Security
    api_key_enabled: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None)
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_calls: int = Field(default=100)
    rate_limit_period: int = Field(default=60)

    # Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/phishing_detector.log")

    # Training
    train_data_path: str = Field(default="data/processed/train.csv")
    test_data_path: str = Field(default="data/processed/test.csv")
    model_save_path: str = Field(default="models/trained")
    random_seed: int = Field(default=42)

    # Project paths
    @property
    def project_root(self) -> Path:
        """Get project root directory"""
        return Path(__file__).parent.parent

    @property
    def models_dir(self) -> Path:
        """Get models directory"""
        return self.project_root / self.model_path

    @property
    def data_dir(self) -> Path:
        """Get data directory"""
        return self.project_root / "data"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
