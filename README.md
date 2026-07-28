# Sales Forecast AI 📈

An end-to-end machine learning and time-series forecasting suite built to train models, evaluate benchmark rankings, generate future forecasts, and extract actionable business insights. Designed for production-grade deployments with a gorgeous, high-fidelity Streamlit dashboard.

---

## 🌟 Features

* **Data Validation & Cleaning**: Automatic validation of schema datatypes, check for missing/null values, duplicate detection, and robust column alignment.
* **Advanced Feature Engineering Pipeline**:
  * **Calendar/DateTime**: Reconstructs year, month, day, day of week, day of year, quarter, week of year, and weekend indicators.
  * **Lag Features**: Auto-shifts target sales history (1, 7, 14, and 28 days) to capture autoregressive behavior.
  * **Rolling Windows**: Computes rolling means and standard deviations across 7-day and 30-day horizons.
* **Unified Model Framework**:
  * Integrates Traditional ML, Deep Learning representation, and Additive Time Series wrappers:
    * Naive Baseline & Linear Regression
    * Random Forest Regressor & XGBoost
    * Additive Prophet Model (with linear fallback)
    * LSTM Recurrent Neural Network (with MLP fallback)
* **Automated Raw Inference Engineering**:
  * When a user uploads a raw dataset (e.g. Kaggle's `test.csv` containing only `date`, `store`, and `item`), the `Predictor` dynamically matches it with historical sales, engineers lag/rolling features, and runs inference instantly without tracebacks.
* **Explainable AI**: Plots feature importances and logs metrics to HTML/Markdown reports.
* **Business Insights & Actionable Recommendations**:
  * **Safety Stock Adjustment**: Suggests inventory buffer increases/decreases based on linear demand trends.
  * **Weekend Staffing**: Pinpoints weekday/weekend sales ratio differences and suggests staffing strategies.
  * **Promo Targeting**: Recommends loyalty campaigns on historically low-volume weekdays.
  * **Peak Demand Warnings**: Flags spike anomalies exceeding 50% of typical sales volume.

---

## 📊 Benchmarking Results (913k Rows)

Models evaluated on the Kaggle *Store Item Demand Forecasting* dataset:

| Rank | Model | RMSE | MAE | R² | Training Time (s) | Inference Time (s) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **1** | **LSTMModel** (MLP Fallback) | **7.9563** | 6.1278 | 0.9366 | 123.33 | 0.568 |
| **2** | **XGBoostModel** | **7.9738** | 6.1349 | 0.9364 | 18.07 | 0.317 |
| **3** | **RandomForestModel** | **8.1698** | 6.2825 | 0.9332 | 791.57 | 107.42 |
| **4** | **LinearRegressionModel** | **9.0346** | 6.9211 | 0.9183 | 0.76 | 0.027 |
| **5** | **ProphetModel** (Linear Fallback) | **29.6874** | 24.3751 | 0.1178 | 0.58 | 0.090 |
| **6** | **NaiveForecastModel** | **47.2587** | 36.4432 | -1.2357 | 0.00 | 0.001 |

---

## 🛠️ Project Structure

```bash
sales-forecast-ai/
├── data/                       # Dataset templates (train.csv is Git-ignored)
│   ├── benchmark_template.csv  # 5-row sample with sales target
│   └── predict_template.csv    # 5-row sample for inference testing
├── notebooks/                  # EDA and experimentation notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_model_benchmarking.ipynb
├── src/                        # Core source code
│   ├── main.py                 # Benchmarking pipeline main entrypoint
│   ├── benchmark.py            # Main pipeline orchestration
│   ├── config/                 # Central configurations & parameters
│   ├── data_pipeline/          # CSV loaders and schema validators
│   ├── experiment/             # Metric loggers and reporting modules
│   ├── explainability/         # Feature importance and SHAP analyzers
│   ├── features/               # Pipeline for datetime, lag, rolling features
│   ├── models/                 # Model registry and base/concrete wrappers
│   ├── prediction/             # Single-step Predictor and multi-step Forecaster
│   ├── tuning/                 # Hyperparameter search frameworks
│   └── utils/                  # Logger, timer, and insights generator
├── tests/                      # Pytest unit-test suite
├── requirements.txt            # Python dependencies
└── streamlit_app.py            # Streamlit dashboard application
```

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/sheetalambre-ai/sales-forecast-ai.git
cd sales-forecast-ai

# Activate virtual environment and install requirements
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Test Suite
Ensure all components are verified:
```bash
$env:PYTHONPATH="src"
pytest
```

### 3. Run the Benchmarking Pipeline
Train all models on local data and output rankings:
```bash
$env:PYTHONPATH="src"
python src/main.py
```

### 4. Start the Streamlit Dashboard
Launch the visualization app:
```bash
$env:PYTHONPATH="src"
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💡 Templates for App Testing
If you run the app inside a sandboxed viewport where file downloads are blocked, you can find premade template CSVs directly in the `data/` folder:
* **Benchmark/EDA**: Upload `data/benchmark_template.csv` or the full `data/raw/train.csv`.
* **Inference**: Upload `data/predict_template.csv` or any Kaggle test set to generate forecasts.