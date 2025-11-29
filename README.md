# Zero-day Phishing Detector (Hybrid Deep Learning)

A state-of-the-art phishing detection system powered by a **Hybrid Deep Learning Architecture**. This model fuses a **Convolutional Neural Network (CNN)** for raw text analysis with a **Multi-Layer Perceptron (MLP)** for engineered feature processing, achieving **99.23% accuracy** and a near-zero false positive rate.

---

## 🚀 Key Features

- **Hybrid Deep Learning Model**: Combines CNN (text patterns) + MLP (statistical features).
- **Dual-Input Analysis**:
  - **Text Branch**: Reads raw URLs to find semantic patterns (e.g., "secure-login").
  - **Feature Branch**: Analyzes 57 explicit features (length, entropy, DNS, etc.).
- **Superior Performance**: 99.23% Accuracy, 0.54% False Positive Rate.
- **Production Ready**: FastAPI-based REST API with real-time inference.
- **Web UI**: Clean, modern interface for testing URLs.

---

## 📊 Proposed System Model

Our system employs a dual-branch neural network architecture that processes the URL from two distinct perspectives before fusing the information for a final decision.

### System Architecture

```mermaid
graph TD
    URL["Input URL"]

    subgraph "Branch A: Feature Engineering (MLP)"
    FE["Extract 57 Features"] --> Scale["Standard Scaler"]
    Scale --> D1["Dense Layer (64 units)"]
    D1 --> D2["Dense Layer (32 units)"]
    end

    subgraph "Branch B: Text Analysis (CNN)"
    Tok["Tokenize URL String"] --> Emb["Embedding Layer"]
    Emb --> CNN1["Conv1D (64 filters)"]
    CNN1 --> Pool["MaxPooling"]
    Pool --> CNN2["Conv1D (128 filters)"]
    CNN2 --> GlobalPool["GlobalMaxPooling"]
    GlobalPool --> D3["Dense Layer (32 units)"]
    end

    D2 --> Concat["Concatenate / Fusion"]
    D3 --> Concat

    Concat --> Final1["Dense Layer (64 units)"]
    Final1 --> Dropout["Dropout (0.4)"]
    Dropout --> Final2["Dense Layer (32 units)"]
    Final2 --> Out["Output Node (Sigmoid)"]
```

### Feature Breakdown (57 Features)

The model utilizes **57 carefully engineered features** to capture both structural and infrastructure-based anomalies:

#### 1. Lexical Features (40)
Derived directly from the URL string:
- **Length**: URL length, hostname length, path length, query length.
- **Counts**: Dots, hyphens, underscores, slashes, special chars (@, &, %, =, ?).
- **Entropy**: Randomness of URL, hostname, and path (detects DGAs).
- **Ratios**: Digit/letter ratio, vowel/consonant ratio, case ratios.
- **Patterns**: IP address usage, "https" token, suspicious TLDs, double slashes.

#### 2. Host-based Features (17)
Derived from external network queries:
- **DNS**: A, MX, NS, TXT, PTR record existence and counts.
- **SSL/TLS**: Certificate validity, age, time-to-expiry, issuer info.
- **Network**: Standard vs. non-standard port usage.

---

## 📈 Performance

The Hybrid Deep Learning model delivers state-of-the-art results, significantly outperforming traditional machine learning approaches.

| Metric | Performance |
| :--- | :--- |
| **Accuracy** | **99.23%** |
| **Precision** | **99.46%** |
| **Recall** | **99.00%** |
| **F1-Score** | **0.9923** |
| **False Positive Rate** | **0.54%** |

---

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

Train the Hybrid Deep Learning model using your dataset:

```bash
python train_hybrid_model.py kaggle.csv
```
*This will save the trained model to `models/hybrid_v1/`.*

### 3. Evaluate the Model

Evaluate the model's performance on a test dataset:

```bash
python evaluate_hybrid.py evaluation_test_data.csv
```

### 4. Start API Server

Start the REST API (loads Hybrid model by default):

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

### 5. Use Web UI

Open `ui.html` in your browser to test URLs through the web interface.

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict single URL |
| `/batch_predict` | POST | Predict multiple URLs |
| `/health` | GET | Health check |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://suspicious-login.com"}'
```

---

## 📁 Project Structure

```
url_phishing_detector/
├── main.py                          # API Entry Point
├── train_hybrid_model.py            # Hybrid Training Script
├── evaluate_hybrid.py               # Hybrid Evaluation Script
├── ui.html                          # Web Interface
├── PROPOSED_SCHEME.md               # Detailed System Specification
├── REPORT.md                        # Final Project Report
├── src/phishing_detector/
│   ├── api.py                       # API Logic
│   ├── detector.py                  # Main Detector Class
│   ├── features/
│   │   ├── extractor.py             # Feature Extractor Orchestrator
│   │   ├── lexical.py               # Lexical Feature Logic
│   │   └── host_features.py         # Host/Network Feature Logic
│   ├── models/
│   │   └── hybrid_model.py          # Hybrid Architecture (CNN+MLP)
│   └── utils/
│       └── url_tokenizer.py         # URL Tokenizer for CNN
└── models/
    └── hybrid_v1/                   # Saved Hybrid Model
```

## 📄 License

MIT License

## 🙏 Acknowledgments

Built with **TensorFlow/Keras** and **FastAPI**.
