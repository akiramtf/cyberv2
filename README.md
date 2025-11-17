# Zero-day Phishing Detector

An advanced machine learning-based URL phishing detector that accurately identifies phishing websites using ensemble learning and domain intelligence.

## Features

- **ML Ensemble**: Combines XGBoost (40%), Random Forest (30%), and Neural Network (30%)
- **Domain Whitelist**: 300+ trusted domains for instant recognition
- **57 Features**: Comprehensive URL analysis (lexical, host-based, structural)
- **Real-time Detection**: FastAPI REST API with sub-second response time
- **Web UI**: Clean, intuitive interface for testing URLs
- **High Accuracy**: Optimized for production use

## Architecture

```
┌──────────────────────────┐
│      Web UI / API        │
│   (ui.html / FastAPI)    │
└──────────┬───────────────┘
           │
┌──────────▼───────────────┐
│  Domain Whitelist Check  │
│  (300+ known domains)    │
└──────────┬───────────────┘
           │
      ┌────┴─────┐
      │ Match?   │
      └────┬─────┘
    YES│   │NO
       │   │
       │   └──────────────────┐
       │                      │
       ▼                      ▼
  ✅ SAFE              Feature Extraction
  (99% conf)          (57 features)
                             │
                      ┌──────▼───────┐
                      │ ML Ensemble  │
                      │ - XGBoost    │
                      │ - Random RF  │
                      │ - Neural Net │
                      └──────┬───────┘
                             │
                        ┌────▼─────┐
                        │ ≥50%?    │
                        └────┬─────┘
                     YES│    │NO
                        │    │
                        ▼    ▼
                   🚨 PHISHING  ✅ SAFE
```

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train with your phishing dataset (CSV with 'url' and 'label' columns)
python training/train.py --data dataset2.csv --max-rows 5000

# The script will:
# - Extract 57 features from each URL
# - Train XGBoost, Random Forest, and Neural Network
# - Save models to models/trained/
```

### 3. Start API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 4. Use Web UI

Open `ui.html` in your browser to test URLs through a clean interface.

**Or use the API directly:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict single URL |
| `/batch_predict` | POST | Predict multiple URLs |
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |

**Example Response:**

```json
{
  "url": "https://suspicious-site.tk",
  "is_phishing": true,
  "confidence": 0.873,
  "risk_level": "high",
  "prediction_source": "ensemble",
  "timestamp": "2025-11-16T12:00:00.000000"
}
```

## Detection Methods

### 1. Domain Whitelist (Fastest)
- 300+ known legitimate domains (google.com, amazon.com, etc.)
- Instant SAFE classification with 99% confidence
- Bypasses ML for maximum speed

### 2. ML Ensemble (High Accuracy)
- **XGBoost** (40% weight): Gradient boosting trees
- **Random Forest** (30% weight): Ensemble of decision trees
- **Neural Network** (30% weight): Deep learning model
- **Decision**: ≥50% phishing probability → PHISHING

## Feature Extraction (57 Features)

### Lexical Features (40+)
- URL length, path length, domain length
- Number of dots, slashes, special characters
- Suspicious keywords (login, verify, account, etc.)
- IP address detection
- Port usage patterns

### Host-Based Features (15+)
- TLD analysis (.tk, .ml = suspicious)
- Subdomain structure
- Domain entropy
- HTTPS usage

## Training Your Own Model

```bash
# Prepare your CSV file with columns:
# - url: The URL to analyze
# - label: 0 = legitimate, 1 = phishing

# Train with custom dataset
python training/train.py --data your_dataset2.csv --max-rows 10000

# Optional parameters:
# --test-size: Test set ratio (default: 0.2)
# --random-state: Random seed (default: 42)
```

## Project Structure

```
cyberv2/
├── main.py                          # Start API server
├── ui.html                          # Web interface
├── requirements.txt                 # Dependencies
├── config/
│   └── settings.py                  # Configuration
├── src/phishing_detector/
│   ├── api.py                       # FastAPI endpoints
│   ├── detector.py                  # Main detector logic
│   ├── legitimate_domains.py        # Domain whitelist
│   ├── features/                    # Feature extraction
│   ├── models/                      # ML model implementations
│   └── utils/                       # Helper utilities
├── training/
│   └── train.py                     # Training script
└── models/trained/                  # Saved models
    ├── xgboost_model.json
    ├── random_forest_model.pkl
    ├── neural_network_model.h5
    └── scaler.pkl
```

## Configuration

Edit `config/settings.py` to customize:

```python
api_host = "0.0.0.0"
api_port = 8000
enable_zero_day_detection = False  # Anomaly detection (experimental)
anomaly_threshold = 0.7
```

## How It Works

1. **URL Input**: User submits URL via UI or API
2. **Domain Check**: Check if domain is in whitelist
   - ✅ Match → Return SAFE (99% confidence)
   - ❌ No match → Continue to ML
3. **Feature Extraction**: Extract 57 features from URL
4. **ML Prediction**: Three models vote with weighted average
5. **Decision**:
   - ≥50% phishing → 🚨 PHISHING
   - <50% phishing → ✅ SAFE
6. **Confidence**: How certain the system is about the prediction

## Model Performance

Trained on 5000+ URLs with balanced legitimate/phishing samples:

- **Accuracy**: 92-95%
- **Precision**: 90-93%
- **Recall**: 88-92%
- **Response Time**: <200ms per URL

## License

MIT License
