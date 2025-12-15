# 🏠 Immo-Eliza Real Estate Price Estimator

Production-ready Streamlit application for Belgian real estate price prediction using specialized XGBoost models.

**Live Demo**: [Deploy on Streamlit Cloud](https://streamlit.io/)

---

## 📋 Overview

This is a clean, minimal ML inference application that provides accurate price predictions for Belgian properties (houses and apartments). It uses dual specialized XGBoost models with confidence intervals, PDF report generation, and a modern dark/light mode UI.

### Key Features

- **Dual-model architecture**: Separate XGBoost models for houses (±16.7% CI) and apartments (±9% CI)
- **Interactive UI**: Modern Streamlit interface with dark/light mode toggle
- **PDF Reports**: Professional valuation reports with property details and location insights
- **Geographic Intelligence**: Postal code-based metadata enrichment with province/region benchmarks
- **Confidence Intervals**: Model-specific uncertainty quantification
- **Production-Ready**: Minimal dependencies, explicit paths, frozen ML logic

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AmineSam/immo-eliza-deployment-Amine_Clean.git
   cd immo-eliza-deployment-Amine_Clean
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app/streamlit_app.py
   ```

5. **Access the app**
   - Open `http://localhost:8501` in your browser

---

## 🏗️ Project Structure

```
immo-eliza-deployment-Amine_Clean/
│
├── app/
│   ├── fonts/
│   │   └── DejaVuSans.ttf          # Unicode font for PDF generation
│   └── streamlit_app.py             # Main Streamlit application
│
├── models/
│   ├── model_xgb_house.pkl          # House XGBoost model (11.1 MB)
│   ├── model_xgb_apartment.pkl      # Apartment XGBoost model (56.0 MB)
│   ├── stage3_pipeline_house.pkl    # House preprocessing pipeline
│   └── stage3_pipeline_apartment.pkl # Apartment preprocessing pipeline
│
├── data/
│   └── pre_processed/
│       └── pre_processed_data_for_kaggle.csv  # Lookup data
│
├── utils/
│   └── stage3_utils.py              # Stage 3 preprocessing
│
├── test_predictions.py              # Behavioral equivalence test
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 📊 Model Information

### House Model
- **Type**: XGBoost Regressor
- **MAE Relative Error**: 16.7%
- **Confidence Interval**: ±16.7%
- **Trained on**: Belgian house data (villas, residences, mixed buildings, cottages, etc.)

### Apartment Model
- **Type**: XGBoost Regressor
- **MAE Relative Error**: 9%
- **Confidence Interval**: ±9%
- **Trained on**: Belgian apartment data (flats, studios, penthouses, duplexes, etc.)

### Features (25 total)

The models use 25 engineered features including:
- Property characteristics (area, rooms, bathrooms, toilets, facades)
- Target-encoded categorical features (postal_code, locality, property_type, property_subtype)
- Geographic benchmarks (province, region, national)
- Economic indicators (median_income)
- Amenities (garage, garden, terrace, pool, equipped kitchen)
- Building condition and year

---

## 🎯 Usage

### Input Fields

**Required**:
- Property Type (House/Apartment)
- Property Subtype (e.g., villa, apartment, penthouse)
- Postal Code (Belgian postal codes)
- Build Year (1800-2030)
- Building State (New/Good/Renovation)

**Adjustable**:
- Living Area (15-500 m²)
- Primary Energy Consumption (0-500 kWh/m²)
- Bedrooms, Bathrooms, Toilets, Facades
- Amenities (Garage, Garden, Terrace, Kitchen, Pool)

### Output

- **Predicted Price**: Estimated property value in EUR
- **Confidence Range**: Lower and upper bounds based on model error
- **Price Benchmarks**: Comparison with province/region averages
- **Location Insights**: Province, region, locality, median income
- **PDF Report**: Downloadable professional valuation report

---

## 🧪 Testing

### Behavioral Equivalence Test

The repository includes a test script to verify prediction consistency:

```bash
# Capture golden outputs (baseline)
python test_predictions.py --capture

# Validate against golden outputs
python test_predictions.py --validate
```

**Test Coverage**:
- House prediction (Brussels villa)
- Apartment prediction (Antwerp)
- House prediction (Ghent residence)

**Success Criteria**: All predictions must match within €1 tolerance.

---

## ☁️ Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Deploy `app/streamlit_app.py`

**Environment Variables**: None required (all paths are relative)

### Local Docker (Optional)

```bash
# Create Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501"]

# Build and run
docker build -t immo-eliza .
docker run -p 8501:8501 immo-eliza
```

---

## 🔧 Technical Details

### Prediction Pipeline

```
User Input
    ↓
Get Metadata (postal_code → locality, province, region, benchmarks)
    ↓
Enrich Input (merge user input + metadata)
    ↓
Select Model (House or Apartment)
    ↓
Stage 3 Preprocessing (missingness flags, imputation, target encoding, log transforms)
    ↓
Feature Selection (25 features in exact order)
    ↓
XGBoost Prediction
    ↓
Confidence Interval (±16.7% for houses, ±9% for apartments)
    ↓
Display Result + PDF Generation
```

### Path Resolution

Uses explicit `pathlib.Path` for robust path resolution:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
model_path = PROJECT_ROOT / "models" / "model_xgb_house.pkl"
```

**Benefits**:
- Works identically locally and on Streamlit Cloud
- No fragile `os.path.dirname(__file__)` chains
- Cross-platform compatible

---

## 📦 Dependencies

Core dependencies (see `requirements.txt`):
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Preprocessing utilities
- `xgboost` - XGBoost models
- `joblib` - Model serialization
- `altair` - Interactive charts
- `fpdf2` - PDF generation

---

## 🤝 Contributing

This is a production deployment of frozen ML models. For model improvements or feature requests, please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

**Note**: Do NOT modify ML logic or preprocessing without re-validating with `test_predictions.py`.

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 👤 Author

**Amine Sam**
- GitHub: [@AmineSam](https://github.com/AmineSam)
- Project: [immo-eliza-deployment-Amine](https://github.com/AmineSam/immo-eliza-deployment-Amine)

---

## 🔮 Future Work

This repository is **frozen** and ready for:
- FastAPI wrapping (REST API endpoints)
- Model retraining with new data
- Feature engineering improvements
- Additional property types (commercial, land, etc.)

**For FastAPI Migration**: See `cleanup_report.md` for prediction contract and feature schema.

---

**Built with ❤️ for the Belgian real estate market**
