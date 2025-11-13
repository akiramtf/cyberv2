"""Main entry point for the API server"""

import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.phishing_detector.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=1 if settings.api_reload else settings.api_workers,
        log_level=settings.log_level.lower(),
    )
