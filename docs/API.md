# API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, the API does not require authentication. For production use, enable API key authentication in `.env`:

```env
API_KEY_ENABLED=true
API_KEY=your-secret-key
```

## Endpoints

### 1. Health Check

Check if the API is running and model is loaded.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "zero_day_enabled": true,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### 2. Predict Single URL

Analyze a single URL for phishing.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "url": "https://example.com"
}
```

**Response:**
```json
{
  "url": "https://example.com",
  "is_phishing": false,
  "confidence": 0.95,
  "risk_level": "safe",
  "prediction_source": "ensemble",
  "zero_day_detected": false,
  "anomaly_score": 0.12,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Risk Levels:**
- `critical`: Confidence >= 0.9 (phishing)
- `high`: Confidence >= 0.7 (phishing)
- `medium`: Confidence < 0.7 (phishing)
- `low`: Confidence > 0.3 (legitimate)
- `safe`: Confidence <= 0.3 (legitimate)

### 3. Batch Predict

Analyze multiple URLs at once (max 100).

**Endpoint:** `POST /batch_predict`

**Request Body:**
```json
{
  "urls": [
    "https://example.com",
    "http://suspicious-site.tk",
    "https://google.com"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "url": "https://example.com",
      "is_phishing": false,
      "confidence": 0.95,
      "risk_level": "safe",
      "prediction_source": "ensemble",
      "zero_day_detected": false,
      "anomaly_score": 0.12,
      "timestamp": "2024-01-15T10:30:00.000Z"
    },
    ...
  ],
  "total_urls": 3,
  "phishing_count": 1,
  "legitimate_count": 2,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### 4. Model Information

Get information about the loaded model.

**Endpoint:** `GET /model/info`

**Response:**
```json
{
  "model_version": "v1",
  "features_count": 57,
  "zero_day_enabled": true,
  "top_features": [
    {"feature": "url_entropy", "importance": 0.085},
    {"feature": "has_suspicious_keyword", "importance": 0.072},
    ...
  ]
}
```

## Example Usage

### cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Batch prediction
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://google.com", "http://suspicious.tk"]}'
```

### Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"url": "https://example.com"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/batch_predict",
    json={"urls": ["https://google.com", "http://suspicious.tk"]}
)
print(response.json())
```

### JavaScript

```javascript
// Single prediction
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({url: 'https://example.com'})
})
.then(response => response.json())
.then(data => console.log(data));
```

## Error Codes

- `400 Bad Request`: Invalid URL format
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error during prediction
- `503 Service Unavailable`: Model not loaded
