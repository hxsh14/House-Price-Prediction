# House Price Prediction

An end-to-end machine learning pipeline that predicts California median house prices
using the **D.A.T.A.** protocol: Discover → Analyze → Train → Assess.

## Tech Stack
| Layer | Library |
|-------|---------|
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| ML | `scikit-learn` |
| Boosting | `xgboost` |
| Config | `PyYAML` |
| Tests | `pytest` |

## Project Structure
```
House-Price-Prediction/
├── data/raw/               # Immutable source CSV (auto-downloaded)
├── data/processed/         # Cleaned & scaled features
├── src/
│   ├── data/ingest.py      # Phase 1 — fetch California Housing dataset
│   ├── features/engineer.py# Phase 2 — clean, engineer, scale, split
│   ├── models/train.py     # Phase 3 — train 6 models with 5-fold CV
│   └── evaluate/metrics.py # Phase 4 — RMSE/MAE/R², plots, final report
├── models/                 # Serialized .joblib models + registry.md
├── reports/eda/            # EDA & evaluation plots (PNG)
├── reports/final_report.md # Auto-generated results summary
├── tests/                  # pytest suite
├── config.yaml             # Seed, paths, hyperparameters
├── requirements.txt
└── main.py                 # Single entrypoint
```

## Models Trained
1. Baseline (median predictor)
2. Linear Regression
3. Ridge Regression
4. Lasso Regression
5. Random Forest
6. XGBoost

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/hxsh14/House-Price-Prediction.git
cd House-Price-Prediction
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# 4. Run tests
pytest tests/ -v

# 5. Launch Streamlit Web Application
streamlit run app.py
```

## Outputs
| Output | Location |
|--------|----------|
| Final report | `reports/final_report.md` |
| EDA plots | `reports/eda/` |
| Model files | `models/*.joblib` |
| Model registry | `models/registry.md` |

## Configuration
Edit `config.yaml` to change the random seed, test split ratio, CV folds,
or model hyperparameters without touching any source code.

## License
MIT
