"""FastAPI application for phishing detection"""

import os
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import validators

from .detector import PhishingDetector
from config.settings import settings

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CyberV2 - URL Phishing Detector API",
    description="Advanced ML-based phishing detection with zero-day capabilities",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector: Optional[PhishingDetector] = None


# Request/Response models
class URLRequest(BaseModel):
    """Single URL prediction request"""

    url: str = Field(..., description="URL to analyze", min_length=1)

    @validator("url")
    def validate_url(cls, v):
        """Validate URL format"""
        # Add scheme if missing
        if not v.startswith(("http://", "https://", "ftp://")):
            v = "http://" + v

        if not validators.url(v):
            raise ValueError("Invalid URL format")
        return v


class BatchURLRequest(BaseModel):
    """Batch URL prediction request"""

    urls: List[str] = Field(..., description="List of URLs to analyze", min_items=1, max_items=100)

    @validator("urls")
    def validate_urls(cls, v):
        """Validate all URLs"""
        validated_urls = []
        for url in v:
            # Add scheme if missing
            if not url.startswith(("http://", "https://", "ftp://")):
                url = "http://" + url

            if not validators.url(url):
                raise ValueError(f"Invalid URL format: {url}")
            validated_urls.append(url)
        return validated_urls


class ModelScores(BaseModel):
    """Individual model scores"""
    xgboost_score: float
    random_forest_score: float
    neural_network_score: float


class PredictionResponse(BaseModel):
    """Prediction response"""

    url: str
    is_phishing: bool
    confidence: float
    risk_level: str
    prediction_source: str
    zero_day_detected: bool
    anomaly_score: float
    model_scores: Optional[ModelScores]  # None for whitelisted domains
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""

    predictions: List[PredictionResponse]
    total_urls: int
    phishing_count: int
    legitimate_count: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    zero_day_enabled: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response"""

    model_version: str
    features_count: int
    zero_day_enabled: bool
    top_features: List[dict]


# Dependency to get detector
def get_detector() -> PhishingDetector:
    """Get detector instance"""
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first.",
        )
    return detector


# API endpoints
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global detector

    try:
        model_path = settings.models_dir
        if os.path.exists(os.path.join(model_path, "xgboost_model.json")):
            logger.info(f"Loading model from {model_path}")
            detector = PhishingDetector(
                enable_zero_day=settings.enable_zero_day_detection,
                enable_dns_lookup=settings.enable_dns_lookup,
                anomaly_threshold=settings.anomaly_threshold,
            )
            detector.load(str(model_path))
            logger.info("Model loaded successfully")
        else:
            logger.warning("No trained model found. Please train the model first.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "CyberV2 - URL Phishing Detector API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if detector is not None else "model_not_loaded",
        model_loaded=detector is not None,
        zero_day_enabled=settings.enable_zero_day_detection if detector else False,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_url(request: URLRequest, det: PhishingDetector = Depends(get_detector)):
    """Predict if a single URL is phishing"""
    try:
        result = det.predict(request.url)
        result["timestamp"] = datetime.utcnow().isoformat()
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error for {request.url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchURLRequest, det: PhishingDetector = Depends(get_detector)
):
    """Predict multiple URLs"""
    try:
        results = det.predict_batch(request.urls)

        # Add timestamps
        for result in results:
            result["timestamp"] = datetime.utcnow().isoformat()

        # Calculate statistics
        phishing_count = sum(1 for r in results if r.get("is_phishing", False))
        legitimate_count = len(results) - phishing_count

        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results],
            total_urls=len(results),
            phishing_count=phishing_count,
            legitimate_count=legitimate_count,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info(det: PhishingDetector = Depends(get_detector)):
    """Get model information"""
    try:
        feature_importance = det.get_feature_importance()
        top_features = [
            {"feature": k, "importance": v} for k, v in list(feature_importance.items())[:10]
        ]

        return ModelInfoResponse(
            model_version=settings.model_version,
            features_count=len(feature_importance),
            zero_day_enabled=settings.enable_zero_day_detection,
            top_features=top_features,
        )
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
