# Zero-day Phishing Detector

A high-performance machine learning-based phishing detection system powered by XGBoost. Analyzes URLs using 57 features including lexical patterns, DNS records, and SSL certificates to identify phishing websites with high accuracy.

## 🚀 Features

- **XGBoost Model**: Fast, accurate gradient boosting classifier
- **57 Features**: Comprehensive URL analysis
  - 39 Lexical features (URL structure, patterns, entropy)
  - 18 Host-based features (DNS, SSL, port analysis)
- **Real-time Detection**: FastAPI REST API with sub-second response
- **Web UI**: Clean, modern interface for testing URLs
- **Batch Processing**: Analyze multiple URLs simultaneously
- **Production Ready**: Optimized for speed and accuracy

## 📊 Architecture

```
┌──────────────────────────┐
│      Web UI / API        │
│   (ui.html / FastAPI)    │
└──────────┬───────────────┘
           │
┌──────────▼───────────────┐
│   Feature Extraction     │
│   (57 features)          │
│                          │
│  Lexical (39):           │
│  • URL structure         │
│  • Special chars         │
│  • Entropy               │
│  • Suspicious keywords   │
│                          │
│  Host-based (18):        │
│  • DNS records (A/MX/NS) │
│  • SSL certificate       │
│  • Port analysis         │
└──────────┬───────────────┘
           │
┌──────────▼───────────────┐
│    XGBoost Classifier    │
│  (Gradient Boosting)     │
└──────────┬───────────────┘
           │
      ┌────▼─────┐
      │  ≥50%?   │
      └────┬─────┘
    YES│   │NO
       │   │
       ▼   ▼
  🚨 PHISHING  ✅ SAFE
```

## 🔧 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd url_phishing_detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train with your dataset (CSV with 'url' and 'label' columns)
python training/train.py --data dataset4.csv

# Optional: Limit training data size
python training/train.py --data dataset4.csv --max-rows 10000

# The script will:
# - Extract 57 features from each URL (with DNS/SSL lookups)
# - Train XGBoost classifier on 100% of data
# - Save model to models/trained/
```

**Note**: Training with DNS/SSL lookups enabled can take several hours for large datasets. For faster training, you can disable them in `training/train.py` (line 835).

### 3. Evaluate the Model

```bash
# Test model on a separate dataset
python evaluate_model.py evaluation_test_data.csv

# Output includes:
# - Accuracy, Precision, Recall, F1-Score
# - False Positive Rate
# - Confusion matrix
# - Misclassified URLs saved to misclassified_urls.csv
```

### 4. Start API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 5. Use Web UI

Open `ui.html` in your browser to test URLs through the web interface.

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict single URL |
| `/batch_predict` | POST | Predict multiple URLs |
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |

### Single URL Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

**Response:**
```json
{
  "url": "https://example.com",
  "is_phishing": false,
  "confidence": 0.123,
  "risk_level": "low",
  "prediction_source": "xgboost",
  "timestamp": "2025-11-21T12:00:00.000000"
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://www.google.com",
      "http://suspicious-site.tk",
      "https://www.apple.com"
    ]
  }'
```

## 📋 Feature Details (57 Total)

### Lexical Features (39)

**Length Features:**
- `url_length`, `hostname_length`, `path_length`, `query_length`
- `domain_length`, `subdomain_length`, `tld_length`

**Character Counts:**
- `num_dots`, `num_hyphens`, `num_underscores`, `num_slashes`
- `num_question_marks`, `num_equal_signs`, `num_at_symbols`
- `num_ampersands`, `num_percent_signs`

**Binary Indicators:**
- `has_ip_address`, `has_port`, `has_https`
- `has_double_slash_in_path`, `has_at_symbol`, `has_hyphen_in_domain`

**Entropy & Ratios:**
- `url_entropy`, `hostname_entropy`, `path_entropy`
- `digit_ratio`, `letter_ratio`, `digit_letter_ratio`
- `uppercase_ratio`, `lowercase_ratio`

**Suspicious Patterns:**
- `is_url_shortener`, `has_suspicious_tld`
- `has_suspicious_keyword`, `num_suspicious_keywords`
- `max_consecutive_digits`, `max_consecutive_letters`

**Query Parameters:**
- `num_query_params`, `has_query_params`

**Domain Structure:**
- `num_subdomains`, `path_depth`

### Host-based Features (18)

**DNS Features (8):**
- `has_dns_a_record`, `num_dns_a_records` - IP address resolution
- `has_dns_mx_record`, `num_dns_mx_records` - Mail server records
- `has_dns_ns_record`, `num_dns_ns_records` - Nameserver records
- `has_dns_txt_record` - TXT records (SPF, DKIM, etc.)
- `has_ptr_record` - Reverse DNS lookup

**SSL Features (7):**
- `has_ssl_cert` - Certificate exists
- `ssl_cert_valid` - Certificate is valid
- `ssl_days_to_expire` - Days until expiration
- `ssl_cert_expires_soon` - Expires in < 30 days
- `ssl_cert_age_days` - Certificate age
- `ssl_cert_is_new` - Issued < 30 days ago
- `ssl_num_san` - Subject Alternative Names count

**Port Features (3):**
- `has_port` - Uses explicit port
- `uses_standard_port` - Uses 80/443
- `uses_non_standard_port` - Uses unusual port

## 🛠️ Utility Scripts

### Check DNS/SSL Features

```bash
# Inspect DNS and SSL features for any URL
python check_dns.py https://www.facebook.com

# Output shows:
# - DNS records (A, MX, NS, TXT, PTR)
# - SSL certificate details
# - Port information
```

### Create Balanced Dataset

```bash
# Create a balanced test dataset
python create_balanced_dataset.py
```

## 📁 Project Structure

```
url_phishing_detector/
├── main.py                          # API server entry point
├── ui.html                          # Web interface
├── requirements.txt                 # Python dependencies
├── config/
│   └── settings.py                  # Configuration settings
├── src/phishing_detector/
│   ├── api.py                       # FastAPI endpoints
│   ├── detector.py                  # Main detector class
│   ├── features/
│   │   ├── extractor.py             # Feature orchestration
│   │   ├── lexical.py               # Lexical features (39)
│   │   └── host_features.py         # DNS/SSL features (18)
│   ├── models/
│   │   └── phishing_model.py        # XGBoost model wrapper
│   └── utils/
│       └── url_parser.py            # URL parsing utilities
├── training/
│   └── train.py                     # Model training script
├── models/trained/                  # Saved models
│   ├── xgboost_model.json           # XGBoost model
│   ├── scaler.pkl                   # Feature scaler
│   └── feature_names.pkl            # Feature name mapping
├── check_dns.py                     # DNS/SSL feature inspector
├── evaluate_model.py                # Model evaluation script
└── create_balanced_dataset.py       # Dataset creation utility
```

## ⚙️ Configuration

Edit `config/settings.py` or create a `.env` file:

```python
# API Configuration
api_host = "0.0.0.0"
api_port = 8000

# Feature Extraction
enable_dns_lookup = True      # Enable DNS lookups (slower but more accurate)
enable_ssl_lookup = True      # Enable SSL checks (follows DNS setting)
enable_whois_lookup = False   # WHOIS lookups (very slow, disabled)
request_timeout = 5           # Timeout for network requests (seconds)

# Model Configuration
model_path = "models/trained"
```

## 🎯 How It Works

1. **URL Input**: User submits URL via UI or API
2. **Feature Extraction**:
   - Parse URL into components
   - Extract 39 lexical features (URL patterns, entropy, etc.)
   - Perform DNS lookups (A, MX, NS, TXT, PTR records)
   - Check SSL certificate (if HTTPS)
   - Calculate 18 host-based features
3. **Prediction**: XGBoost model analyzes all 57 features
4. **Decision**:
   - Probability ≥ 50% → 🚨 **PHISHING**
   - Probability < 50% → ✅ **SAFE**
5. **Response**: Return prediction with confidence score

## 📈 Model Performance

Performance on balanced test dataset (1000 legitimate + 1000 phishing URLs):

- **Accuracy**: ~94%
- **Precision**: ~93%
- **Recall**: ~95%
- **F1-Score**: ~0.94
- **False Positive Rate**: ~6%
- **Response Time**: <200ms per URL (without DNS/SSL), ~2-5s (with DNS/SSL)

## 🔍 Training vs Prediction

### Training Mode (train.py)
- **DNS/SSL Lookups**: ENABLED (line 835: `enable_dns_lookup=True`)
- **Features Used**: All 57 features
- **Speed**: Slow (network requests for each URL)
- **Purpose**: Learn from real DNS/SSL patterns

### Prediction Mode (api.py)
- **DNS/SSL Lookups**: ENABLED by default (`settings.enable_dns_lookup=True`)
- **Features Used**: All 57 features
- **Speed**: Slower but more accurate
- **Purpose**: Real-time phishing detection

**Note**: You can disable DNS/SSL lookups in production for faster predictions (lexical features only), but accuracy may decrease slightly.

## 📝 Dataset Format

Your training/testing CSV should have two columns:

```csv
url,label
https://www.google.com,0
http://phishing-site.tk,1
https://www.apple.com,0
http://fake-paypal.ml,1
```

- `url`: The URL to analyze
- `label`: 0 = legitimate, 1 = phishing

## 🚀 Performance Tips

1. **Fast Training**: Disable DNS/SSL lookups in `train.py` (set `enable_dns_lookup=False`)
2. **Fast Prediction**: Disable DNS/SSL in `config/settings.py` (set `enable_dns_lookup=False`)
3. **Balanced Dataset**: Use equal numbers of legitimate and phishing URLs
4. **Large Dataset**: More training data = better accuracy (recommended: 10K+ URLs)

## 🐛 Troubleshooting

### SSL Warning
If you see `NotOpenSSLWarning`, it's already suppressed in the code. The system works fine with LibreSSL.

### DNS Lookup Failures
Some URLs may fail DNS lookups (timeouts, no records). The system handles this gracefully by using default values (0.0).

### Model Not Found
Run `python training/train.py --data <your_dataset.csv>` to train the model first.

## 📄 License

MIT License

## 🙏 Acknowledgments

Built with:
- **XGBoost** - Gradient boosting framework
- **FastAPI** - Modern web framework
- **scikit-learn** - Machine learning utilities
- **dnspython** - DNS toolkit
- **tldextract** - Domain extraction
