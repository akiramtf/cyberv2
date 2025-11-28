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
    description="Advanced ML-based phishing detection using ensemble classification",
    version="2.0.0",
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

        # Allow localhost and private IPs for testing
        if "localhost" in v or "127.0.0.1" in v:
            return v

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

            # Allow localhost and private IPs for testing
            if "localhost" in url or "127.0.0.1" in url:
                validated_urls.append(url)
                continue

            if not validators.url(url):
                raise ValueError(f"Invalid URL format: {url}")
            validated_urls.append(url)
        return validated_urls





class PredictionResponse(BaseModel):
    """Prediction response"""

    url: str
    is_phishing: bool
    phishing_score: float
    risk_level: str
    prediction_source: str
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
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response"""

    model_version: str
    features_count: int
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
        model_type = os.environ.get("PHISHING_MODEL_TYPE", "hybrid")
        
        logger.info(f"Initializing {model_type} model from {model_path}")
        
        detector = PhishingDetector(
            model_type=model_type,
            enable_dns_lookup=settings.enable_dns_lookup,
        )
        
        # Check if model files exist before loading
        if model_type == "hybrid":
            # Check for hybrid model files
            # 1. In models_dir (e.g. models/trained)
            path1 = os.path.join(model_path, "hybrid_model.keras")
            # 2. In models_dir/hybrid_v1
            path2 = os.path.join(model_path, "hybrid_v1", "hybrid_model.keras")
            # 3. In project_root/models/hybrid_v1 (Fallback)
            path3 = os.path.join(settings.project_root, "models", "hybrid_v1", "hybrid_model.keras")
            
            if os.path.exists(path1):
                detector.load(str(model_path))
                logger.info(f"{model_type.capitalize()} model loaded successfully from {model_path}")
            elif os.path.exists(path2):
                detector.load(str(model_path))
                logger.info(f"{model_type.capitalize()} model loaded successfully from {model_path}")
            elif os.path.exists(path3):
                # Load from the specific directory found
                load_dir = os.path.dirname(path3)
                detector.load(load_dir)
                logger.info(f"{model_type.capitalize()} model loaded successfully from {load_dir}")
            else:
                logger.warning(f"No trained {model_type} model found. Checked: {path1}, {path2}, {path3}")
        else:
            # Check for xgboost model file
            if os.path.exists(os.path.join(model_path, "xgboost_model.json")):
                detector.load(str(model_path))
                logger.info(f"{model_type.capitalize()} model loaded successfully")
            else:
                logger.warning(f"No trained {model_type} model found at {model_path}. Please train it first.")
                
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
