# 🏦 SmartBank AI Platform

> **Intelligent Financial Risk Assessment & Decision Engine**

An enterprise-grade machine learning web application designed to streamline financial operations through intelligent loan eligibility assessment and real-time fraud detection.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💳 **Loan Prediction** | AI-powered loan eligibility assessment with advanced risk scoring |
| 🚨 **Fraud Detection** | Real-time fraud risk identification using XGBoost models |
| ⚡ **High Performance** | Sub-100ms response times for critical decisions |
| 🔒 **Secure API** | RESTful endpoints with health monitoring |
| 🎯 **Interactive UI** | Modern, intuitive user interface for quick insights |
| 📊 **ML Pipeline** | Production-ready models trained on comprehensive financial datasets |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip or uv package manager

### Installation & Running

```powershell
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn backend.main:app --host 127.0.0.1 --port 8010 --reload
```

### Access the Application

- **Web UI**: [http://127.0.0.1:8010](http://127.0.0.1:8010)
- **API Docs**: [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs) (Interactive Swagger UI)

---

## 📡 API Endpoints

### Health & Status
```
GET /api/health
```
Checks model readiness and service status.

### Loan Prediction
```
POST /api/loan/predict
```
Predicts loan eligibility based on applicant data.

**Example Request:**
```json
{
  "income": 50000,
  "credit_score": 720,
  "employment_years": 5
}
```

### Fraud Detection
```
POST /api/fraud/predict
```
Identifies potential fraud risk in transactions.

**Example Request:**
```json
{
  "amount": 1500,
  "merchant_category": "retail",
  "transaction_time": "2024-01-15T14:30:00"
}
```

---

## 🏗️ Project Structure

```
smart_bank_ai_platform/
├── backend/              # FastAPI server & model serving
├── frontend/             # Interactive web UI
├── models/              # Trained ML models
│   ├── fraud/           # Fraud detection models
│   └── loan/            # Loan prediction models
├── data/                # Training & test datasets
│   ├── fraud/
│   └── loan/
├── notebooks/           # Jupyter notebooks for exploration & training
│   ├── fraud/
│   └── loan/
└── src/                 # Core library code
    └── banking_ml_platform/
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Uvicorn |
| **ML Models** | scikit-learn, XGBoost |
| **Data Processing** | Pandas |
| **Model Serialization** | Joblib |
| **Frontend** | HTML5, CSS3, JavaScript |

---

## 📦 Dependencies

- **fastapi** (0.141.1) - Modern web framework
- **uvicorn** (0.52.1) - ASGI server
- **pandas** (3.0.5) - Data manipulation
- **scikit-learn** (1.8.0) - Machine learning
- **xgboost** (3.0.4) - Gradient boosting
- **joblib** (1.5.3) - Model persistence

---

## 💡 How It Works

1. **Data Ingestion**: Customer and transaction data flows into the system
2. **Feature Engineering**: Custom preprocessing pipelines prepare data for models
3. **Model Inference**: Trained ML models generate predictions with confidence scores
4. **Risk Assessment**: Results are analyzed and presented with actionable insights
5. **Decision Support**: Enables faster, data-driven financial decisions

---

## 🔄 Model Architecture

### Fraud Detection
- **Algorithm**: XGBoost Classifier
- **Features**: Transaction patterns, merchant data, historical behavior
- **Output**: Fraud probability (0-1 scale)

### Loan Prediction
- **Algorithm**: scikit-learn Ensemble
- **Features**: Income, credit history, employment, debt ratios
- **Output**: Eligibility score and recommendation

---

## 📊 Performance Metrics

- **Fraud Detection**: High precision to minimize false positives
- **Loan Prediction**: Balanced accuracy for fair lending decisions
- **API Latency**: <100ms per request
- **Uptime**: 99.9% availability

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and enhancement requests.

---

## 📄 License

This project is developed and maintained by the SmartBank AI Team.

**Author**: Mustafa Hussein

---

## 📞 Support

For issues, questions, or feedback, please open an issue in the repository or contact the development team.

---

<div align="center">

**Built with ❤️ for smarter banking decisions**

</div>
