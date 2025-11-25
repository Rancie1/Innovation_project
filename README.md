# AI-Powered Vulnerability Detection System

A full-stack machine learning application for detecting security vulnerabilities in code snippets. This project combines machine learning model training (Assignment 2) with a production-ready web application (Assignment 3) featuring an interactive React frontend and FastAPI backend.

## 🎯 Project Overview

This system provides two types of vulnerability classification:

1. **Severity Type Classification** - Multi-class classification that identifies specific Common Weakness Enumeration (CWE) categories (e.g., CWE-79, CWE-89, CWE-22)
2. **Binary Classification** - Simple binary classification that determines if code is "Safe" or "Unsafe"

The application features:
- Interactive web interface with real-time predictions
- Multiple ML models (Logistic Regression, Random Forest) for both classification types
- Rich data visualizations (confidence gauges, probability distributions, pie charts, bar charts)
- Model switching and classification type selection
- Dark/light theme support
- Responsive design

## 📁 Project Structure

```
Innovation_project/
├── Assignment2/              # Machine Learning Model Training
│   ├── model-1/              # Binary classification models
│   │   ├── model_1.py        # Training pipeline for binary classification
│   │   └── data/             # Training datasets
│   ├── model-2/              # Multi-class CWE classification models
│   │   ├── model_2.py        # Main training script
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── preprocessor-2.py
│   │   └── data/             # CWE-labeled datasets
│   └── model-3/              # CVSS severity regression models
│       ├── model-3.py
│       └── data/
│
└── Assignment3/             # Full-Stack Web Application
    ├── backend/             # FastAPI REST API
    │   ├── main.py          # API routes and server
    │   ├── model_service.py # Model management and prediction
    │   ├── model_loader.py  # Loads Assignment2 Model-2 models
    │   ├── model1_loader.py  # Loads Assignment2 Model-1 models
    │   ├── models/          # Saved model artifacts (.pkl files)
    │   └── requirements.txt
    │
    └── frontend/            # React Web Application
        ├── src/
        │   ├── App.js       # Main application component
        │   ├── components/  # React components
        │   │   ├── VulnerabilityDashboard.js  # Visualization dashboard
        │   │   ├── CodeInputForm.js
        │   │   └── ThemeToggle.js
        │   └── services/    # API client services
        └── package.json
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with pip
- Node.js 14+ and npm
- Virtual environment support (venv)

### 1. Backend Setup

```bash
cd Assignment3/backend
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Models (if not already trained)

The models from Assignment 2 need to be trained and saved:

```bash
# Train multi-class CWE classification models
python model_loader.py

# Train binary classification models
python model1_loader.py
```

Model artifacts will be saved in `backend/models/` directory.

### 3. Start Backend Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 4. Frontend Setup

```bash
cd Assignment3/frontend
npm install
npm start
```

The application will open at http://localhost:3000

## 🔧 Key Features

### Frontend Features
- **Classification Type Selector**: Switch between Severity Type and Binary classification
- **Model Selection**: Choose between Logistic Regression and Random Forest models
- **Interactive Visualizations**:
  - Confidence gauge showing prediction confidence
  - Probability bar charts for all categories/classes
  - Pie charts showing distribution
  - Prediction history tracking
- **Theme Support**: Dark/light mode with persistent preferences
- **Info Modals**: Detailed descriptions of each classification type
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Error Handling**: User-friendly error messages and loading states

### Backend Features
- **RESTful API**: Clean REST endpoints for model management and predictions
- **Dynamic Model Loading**: Models loaded on-demand based on selection
- **Multiple Model Support**: Seamlessly switches between different model types
- **Response Formatting**: Automatically formats predictions based on model type
- **CORS Enabled**: Configured for frontend integration
- **Health Monitoring**: Health check endpoint for system monitoring

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check endpoint |
| GET | `/models` | List available models and current selection |
| PUT | `/model` | Switch active model |
| POST | `/predict` | Get vulnerability prediction for code snippet |

See [Assignment3/README.md](Assignment3/README.md) for detailed API documentation and examples.

## 🧠 Machine Learning Models

### Model Types

1. **Severity Type Models** (Multi-class Classification)
   - **Logistic Regression**: Fast, interpretable model for CWE category prediction
   - **Random Forest**: Ensemble method for improved accuracy on complex patterns
   - **Output**: Specific CWE categories with probability distributions

2. **Binary Classification Models**
   - **Logistic Regression**: Binary classifier for Safe/Unsafe detection
   - **Random Forest**: Ensemble binary classifier
   - **Output**: Safe or Unsafe classification with confidence scores

### Model Training (Assignment 2)

Models are trained using:
- **TF-IDF Vectorization**: Text feature extraction (max_features=5000)
- **Scikit-learn**: Machine learning framework
- **Pickle Serialization**: Model persistence for production use

For detailed training procedures, see the individual model files in `Assignment2/`.

## 🎨 Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Scikit-learn**: Machine learning library
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server
- **NumPy/Pandas**: Data processing

### Frontend
- **React**: UI framework
- **Recharts**: Data visualization library
- **Axios**: HTTP client
- **CSS Variables**: Theme system

## 📚 Documentation

- **[Assignment3/README.md](Assignment3/README.md)** - Comprehensive documentation for the full-stack application, including:
  - Detailed setup instructions
  - API endpoint documentation
  - Architecture overview
  - Troubleshooting guide
  - Feature descriptions

- **[Assignment2/README.md](Assignment2/README.md)** - Machine learning model training documentation

## 🐛 Troubleshooting

### Common Issues

1. **Models not loading**: Ensure models are trained first using `model_loader.py` and `model1_loader.py`
2. **CORS errors**: Backend is configured for `http://localhost:3000` by default
3. **Version conflicts**: Ensure scikit-learn version matches between training and inference (see `requirements.txt`)
4. **Port conflicts**: Change ports in `uvicorn` command or frontend `.env` file

For more detailed troubleshooting, see [Assignment3/README.md](Assignment3/README.md#troubleshooting).

## 🔮 Future Enhancements

Potential improvements for the system:
- User authentication and authorization
- Database integration for prediction history
- Batch prediction API
- Model versioning system
- Real-time model retraining
- Additional visualization options
- Docker containerization
- CI/CD pipeline
- Unit and integration tests
- Performance metrics dashboard

## 📝 License

Built by Nathan Rancie as part of the Innovation Project.

## 🤝 Contributing

This is an academic project. For questions or improvements, please refer to the assignment guidelines.

---

**Note**: For detailed setup, API documentation, and advanced usage, please refer to [Assignment3/README.md](Assignment3/README.md).
