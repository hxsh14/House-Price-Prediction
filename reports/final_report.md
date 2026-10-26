# House Price Prediction — Final Report

**Best Model:** `xgboost` | **Test RMSE:** `0.1424`

## Test Set Results

| Model             |     RMSE |       MAE |          R2 |   CV RMSE |
|:------------------|---------:|----------:|------------:|----------:|
| xgboost           | 0.142412 | 0.0997879 |  0.839094   |  0.142495 |
| random_forest     | 0.161102 | 0.113773  |  0.794088   |  0.160154 |
| ridge             | 0.214158 | 0.160331  |  0.636132   |  0.208628 |
| linear_regression | 0.21416  | 0.160326  |  0.636124   |  0.208629 |
| lasso             | 0.281593 | 0.225354  |  0.370901   |  0.280244 |
| baseline          | 0.355778 | 0.287937  | -0.00423077 |  0.357579 |

## Notes
- Target: `log1p(MedHouseVal)` — exponentiate predictions with `expm1()`.
- Scaler saved to `models/scaler.joblib`.
- All plots saved to `reports/eda/`.
