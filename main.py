"""Main entry point for the API server"""

import uvicorn
import argparse
import os
from config.settings import settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Phishing Detector API")
    parser.add_argument(
        "--model", 
        type=str, 
        default="hybrid", 
        choices=["hybrid", "xgboost"],
        help="Model type to use (hybrid or xgboost)"
    )
    args = parser.parse_args()
    
    # Set environment variable for API to pick up
    os.environ["PHISHING_MODEL_TYPE"] = args.model
    
    uvicorn.run(
        "src.phishing_detector.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=1 if settings.api_reload else settings.api_workers,
        log_level=settings.log_level.lower(),
    )
