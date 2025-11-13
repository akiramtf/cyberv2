"""Integration tests for API endpoints"""

import pytest
from fastapi.testclient import TestClient
from src.phishing_detector.api import app


client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_predict_endpoint_no_model(self):
        """Test predict endpoint when model is not loaded"""
        response = client.post("/predict", json={"url": "https://example.com"})
        # Should return 503 if model not loaded
        assert response.status_code in [200, 503]

    def test_predict_invalid_url(self):
        """Test predict endpoint with invalid URL"""
        response = client.post("/predict", json={"url": "not-a-url"})
        # Should return 400 for invalid URL or 503 if model not loaded
        assert response.status_code in [400, 422, 503]

    def test_batch_predict_endpoint(self):
        """Test batch predict endpoint"""
        urls = ["https://google.com", "https://example.com"]
        response = client.post("/batch_predict", json={"urls": urls})
        # Should return 200 if model loaded, 503 otherwise
        assert response.status_code in [200, 503]

    def test_batch_predict_too_many_urls(self):
        """Test batch predict with too many URLs"""
        urls = [f"https://example{i}.com" for i in range(101)]
        response = client.post("/batch_predict", json={"urls": urls})
        # Should return 422 for validation error
        assert response.status_code == 422

    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        # Should return 200 if model loaded, 503 otherwise
        assert response.status_code in [200, 503]
