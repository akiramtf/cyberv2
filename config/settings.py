"""Configuration settings for the phishing detector"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    api_reload: bool = Field(default=False, env="API_RELOAD")

    # Model Configuration
    model_path: str = Field(default="models/trained", env="MODEL_PATH")
    model_version: str = Field(default="v1", env="MODEL_VERSION")
    enable_zero_day_detection: bool = Field(default=True, env="ENABLE_ZERO_DAY_DETECTION")
    anomaly_threshold: float = Field(default=0.3, env="ANOMALY_THRESHOLD")

    # Feature Extraction
    enable_dns_lookup: bool = Field(default=True, env="ENABLE_DNS_LOOKUP")
    enable_whois_lookup: bool = Field(default=False, env="ENABLE_WHOIS_LOOKUP")
    enable_content_fetch: bool = Field(default=False, env="ENABLE_CONTENT_FETCH")
    request_timeout: int = Field(default=5, env="REQUEST_TIMEOUT")

    # Security
    api_key_enabled: bool = Field(default=False, env="API_KEY_ENABLED")
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_calls: int = Field(default=100, env="RATE_LIMIT_CALLS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/phishing_detector.log", env="LOG_FILE")

    # Training
    train_data_path: str = Field(default="data/processed/train.csv", env="TRAIN_DATA_PATH")
    test_data_path: str = Field(default="data/processed/test.csv", env="TEST_DATA_PATH")
    model_save_path: str = Field(default="models/trained", env="MODEL_SAVE_PATH")
    random_seed: int = Field(default=42, env="RANDOM_SEED")

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
