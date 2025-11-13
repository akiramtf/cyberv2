# Usage Guide

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cyberv2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment (optional):
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Training the Model

### Using Sample Data

Train the model with built-in sample data:

```bash
python training/train.py
```

### Using Your Own Data

Prepare a CSV file with two columns: `url` and `label` (0 = legitimate, 1 = phishing).

```csv
url,label
https://google.com,0
http://phishing-site.tk,1
```

Train the model:

```bash
python training/train.py --data path/to/your/data.csv
```

### Training Options

```bash
python training/train.py \
  --data data/training_data.csv \
  --output models/trained \
  --test-size 0.2 \
  --random-state 42
```

## Running the API Server

Start the API server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

View the interactive API documentation at `http://localhost:8000/docs`

## Using the Detector Programmatically

### Basic Usage

```python
from src.phishing_detector.detector import PhishingDetector

# Initialize detector
detector = PhishingDetector(
    enable_zero_day=True,
    enable_dns_lookup=True,
    anomaly_threshold=0.3
)

# Load trained model
detector.load("models/trained")

# Predict single URL
result = detector.predict("https://example.com")
print(f"Is phishing: {result['is_phishing']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Risk level: {result['risk_level']}")
```

### Batch Predictions

```python
urls = [
    "https://google.com",
    "http://suspicious-site.tk",
    "https://github.com"
]

results = detector.predict_batch(urls)

for result in results:
    print(f"{result['url']}: {result['risk_level']}")
```

### Feature Extraction Only

```python
from src.phishing_detector.features.extractor import FeatureExtractor

extractor = FeatureExtractor(enable_dns_lookup=True)

# Extract features from single URL
features = extractor.extract_features("https://example.com")
print(f"Extracted {len(features)} features")

# Extract features from multiple URLs
import pandas as pd
urls = ["https://google.com", "https://example.com"]
df = extractor.extract_batch(urls)
print(df.head())
```

## Configuration

Edit `.env` to configure the detector:

```env
# Enable/disable zero-day detection
ENABLE_ZERO_DAY_DETECTION=true

# Anomaly detection threshold (0-1, higher = more sensitive)
ANOMALY_THRESHOLD=0.3

# Enable DNS lookups (more accurate but slower)
ENABLE_DNS_LOOKUP=true

# Request timeout for DNS/SSL checks
REQUEST_TIMEOUT=5

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

## Advanced Usage

### Training with Custom Parameters

```python
import numpy as np
from src.phishing_detector.detector import PhishingDetector

# Prepare your data
X_train = np.array(...)  # Feature matrix
y_train = np.array(...)  # Labels (0 or 1)

# Initialize and train
detector = PhishingDetector(
    enable_zero_day=True,
    anomaly_threshold=0.3,
    random_state=42
)

history = detector.train(X_train, y_train)

# Save model
detector.save("models/custom_model")
```

### Feature Importance Analysis

```python
# Get feature importance
importance = detector.get_feature_importance()

# Print top 10 features
for feature, score in list(importance.items())[:10]:
    print(f"{feature}: {score:.4f}")
```

### Zero-Day Detection Tuning

```python
# Adjust zero-day detection sensitivity
detector = PhishingDetector(
    enable_zero_day=True,
    anomaly_threshold=0.5  # Higher = more sensitive
)

# Check anomaly score for suspicious URLs
result = detector.predict("http://new-phishing-site.xyz")
print(f"Anomaly score: {result['anomaly_score']:.2f}")
print(f"Zero-day detected: {result['zero_day_detected']}")
```

## Testing

Run unit tests:

```bash
pytest tests/unit/
```

Run integration tests:

```bash
pytest tests/integration/
```

Run all tests with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Docker Deployment (Coming Soon)

```bash
docker build -t cyberv2-detector .
docker run -p 8000:8000 cyberv2-detector
```

## Performance Tips

1. **Disable DNS lookups for faster inference:**
   ```python
   detector = PhishingDetector(enable_dns_lookup=False)
   ```

2. **Use batch predictions for multiple URLs:**
   ```python
   results = detector.predict_batch(urls)  # More efficient
   ```

3. **Adjust worker count for API:**
   ```env
   API_WORKERS=8  # Increase for better throughput
   ```

4. **Pre-load model at startup:**
   The API automatically loads the model on startup for faster first predictions.
