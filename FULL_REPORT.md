# Zero-day Phishing Detection Using Hybrid Deep Learning Architecture
**Authors**: Phishing Detection Research Team  
**Date**: November 28, 2025  
**Version**: 3.0 (Final)

---

## Abstract

Phishing attacks continue to pose significant cybersecurity threats, with attackers constantly evolving their techniques to bypass traditional detection methods. This paper presents a novel **Hybrid Deep Learning Model** for phishing detection that significantly outperforms traditional machine learning approaches. Our proposed architecture fuses a **Multi-Layer Perceptron (MLP)** for processing 57 engineered features with a **1D Convolutional Neural Network (CNN)** for analyzing raw URL text sequences. This dual-input approach allows the model to leverage both statistical domain knowledge and learned textual patterns simultaneously. Evaluation on a balanced dataset of 10,000 URLs demonstrates state-of-the-art performance, achieving **99.23% accuracy** and a **False Positive Rate of only 0.54%**, compared to 94.5% accuracy and 6.2% FPR for a baseline XGBoost model. The system is deployed as a production-ready REST API, offering robust protection against zero-day phishing threats.

**Keywords**: Phishing Detection, Hybrid Deep Learning, CNN, MLP, Cybersecurity, URL Analysis

---

## I. Introduction

### A. Background and Motivation

Phishing attacks remain a dominant vector for cybercrime, exploiting human trust to steal credentials and financial data. While traditional blacklist-based methods are effective against known threats, they fail to detect newly created "zero-day" phishing sites. Machine learning offers a solution by identifying patterns in URLs, but most existing approaches rely either solely on manual feature engineering (which can miss subtle textual patterns) or solely on deep learning (which may ignore explicit domain knowledge).

### B. Problem Statement

The core challenge is to design a model that:
1.  **Understands Explicit Features**: Captures known indicators like URL length, dot count, and IP address usage.
2.  **Learns Implicit Patterns**: Automatically detects suspicious keyword sequences (e.g., "secure-login", "apple-id") without manual rule creation.
3.  **Minimizes False Positives**: Ensures legitimate business sites are not blocked, a critical requirement for user acceptance.

### C. Our Contribution

We propose a **Hybrid Deep Learning Architecture** that combines the strengths of feature engineering and deep learning. Our system:
1.  **Integrates Two Neural Networks**: A Dense Network for tabular features and a CNN for text analysis.
2.  **Achieves Superior Accuracy**: 99.23% accuracy, reducing the error rate by over 85% compared to XGBoost.
3.  **Provides Production Readiness**: A scalable API capable of real-time inference.

---

## II. Proposed Scheme: Hybrid Deep Learning Model

### A. System Architecture

Our model employs a dual-branch architecture that processes the URL from two distinct perspectives before fusing the information for a final decision.

```mermaid
graph TD
    URL["Input URL"]

    subgraph "Branch A: Feature Engineering (MLP)"
    FE[Extract 57 Features] --> Scale[Standard Scaler]
    Scale --> D1[Dense Layer (64 units, ReLU)]
    D1 --> D2[Dense Layer (32 units, ReLU)]
    end

    subgraph "Branch B: Text Analysis (CNN)"
    Tok[Tokenize URL String] --> Emb[Embedding Layer]
    Emb --> CNN1[Conv1D (64 filters)]
    CNN1 --> Pool[MaxPooling]
    Pool --> CNN2[Conv1D (128 filters)]
    CNN2 --> GlobalPool[Global MaxPooling]
    GlobalPool --> D3[Dense Layer (32 units, ReLU)]
    end

    D2 --> Concat[Concatenate / Fusion Layer]
    D3 --> Concat

    Concat --> Final1[Dense Layer (64 units, ReLU)]
    Final1 --> Dropout[Dropout (0.4)]
    Dropout --> Final2[Dense Layer (32 units, ReLU)]
    Final2 --> Out[Output Node (Sigmoid)]
```

### B. Branch A: Tabular Features (The "Analyst")

This branch processes explicit statistical features extracted from the URL.
*   **Input**: A vector of 57 numerical features (40 Lexical + 17 Host-based).
*   **Processing**: A Multi-Layer Perceptron (MLP) learns non-linear relationships between these features.
*   **Key Features**:
    *   **Lexical**: URL length, dot count, special character ratios, entropy.
    *   **Structural**: Path depth, query parameter count.
    *   **Host-based**: DNS records, SSL validity, port checks.

### C. Branch B: Text Features (The "Linguist")

This branch treats the URL as a sequence of characters/words to find semantic patterns.
*   **Input**: The raw URL string, tokenized into integer sequences.
*   **Processing**:
    *   **Embedding**: Maps characters/sub-words to dense vectors.
    *   **Convolutional Layers (CNN)**: Slides filters over the text to detect local patterns (e.g., "paypal" followed by "update").
    *   **Pooling**: Extracts the most salient features from the text map.

### D. Fusion and Classification

The high-level representations from both branches are concatenated. This "Hybrid" vector contains both the statistical summary and the textual semantic understanding. Final dense layers process this combined signal to output a probability score ($0.0$ to $1.0$).

---

## III. Evaluation

### A. Dataset

*   **Source**: Public Phishing Dataset (e.g., Kaggle/PhishTank).
*   **Size**: 10,000 URLs (Balanced).
*   **Split**: 50% Legitimate, 50% Phishing.

### B. Experimental Results

We compared our proposed Hybrid model against a strong baseline XGBoost model (which uses only tabular features).

| Metric | Baseline (XGBoost) | **Proposed (Hybrid DL)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 94.50% | **99.23%** | **+4.73%** |
| **Precision** | 93.20% | **99.46%** | **+6.26%** |
| **Recall** | 95.80% | **99.00%** | **+3.20%** |
| **F1-Score** | 0.9448 | **0.9923** | **+0.0475** |
| **ROC-AUC** | 0.9612 | **0.9979** | **+0.0367** |
| **False Positive Rate** | 6.20% | **0.54%** | **-91.3% (Critical)** |

### C. Analysis

1.  **Drastic Reduction in False Positives**: The most significant improvement is the reduction of the False Positive Rate from 6.2% to **0.54%**. This means the Hybrid model rarely blocks legitimate sites, solving a major usability pain point of the baseline model.
2.  **Deep Pattern Recognition**: The CNN branch successfully identifies phishing URLs that mimic legitimate structures (e.g., `apple-support-secure.com`) which might look statistically "normal" to the XGBoost model but contain suspicious keyword sequences.
3.  **Robustness**: The model is extremely confident, often outputting scores near $0.0$ or $1.0$, indicating a clear separation capability.

---

## IV. Conclusion

This research demonstrates that a **Hybrid Deep Learning Model**—combining explicit feature engineering with implicit text pattern learning—is superior to traditional single-method approaches for phishing detection. By fusing an MLP with a CNN, we achieved **99.23% accuracy** and virtually eliminated false positives. This architecture provides a robust, high-performance solution for real-time zero-day phishing detection.

---

## Appendix: Implementation Details

*   **Framework**: TensorFlow/Keras
*   **Optimizer**: Adam (Learning Rate: 0.001)
*   **Loss Function**: Binary Cross-Entropy
*   **Training**: 20 Epochs with Early Stopping
*   **Infrastructure**: Python 3.9, FastAPI for deployment
