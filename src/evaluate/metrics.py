"""Evaluation — metrics, plots, and final report generation."""
import os
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def plot_predictions(y_true, y_pred, model_name: str, eda_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.3, color="steelblue", s=8)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5)
    axes[0].set_xlabel("Actual (log scale)")
    axes[0].set_ylabel("Predicted (log scale)")
    axes[0].set_title(f"{model_name} — Predicted vs Actual")

    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Residual")
    axes[1].set_title(f"{model_name} — Residuals")

    plt.tight_layout()
    os.makedirs(eda_dir, exist_ok=True)
    plt.savefig(os.path.join(eda_dir, f"{model_name}_eval.png"), dpi=100)
    plt.close()


def plot_feature_importance(model, feature_names, model_name: str, eda_dir: str) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    imp  = model.feature_importances_
    idx  = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(imp)), imp[idx], color="steelblue")
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
    ax.set_title(f"{model_name} — Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, f"{model_name}_importance.png"), dpi=100)
    plt.close()


def plot_model_comparison(rows: list, eda_dir: str) -> None:
    df = pd.DataFrame(rows).sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["gold" if i == 0 else "steelblue" for i in range(len(df))]
    ax.barh(df["Model"], df["RMSE"], color=colors)
    ax.set_xlabel("Test RMSE (lower is better)")
    ax.set_title("Model Comparison — Test RMSE")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "model_comparison.png"), dpi=100)
    plt.close()


def evaluate(
    train_results: dict,
    X_test, y_test,
    feature_names,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    config      = load_config(config_path)
    eda_dir     = config["paths"]["eda_dir"]
    reports_dir = config["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    rows = []
    best_name, best_rmse = None, float("inf")

    for name, result in train_results.items():
        model  = result["model"]
        y_pred = model.predict(X_test)
        m      = compute_metrics(y_test, y_pred)

        plot_predictions(y_test, y_pred, name, eda_dir)
        plot_feature_importance(model, feature_names, name, eda_dir)

        rows.append({
            "Model":   name,
            "RMSE":    m["RMSE"],
            "MAE":     m["MAE"],
            "R2":      m["R2"],
            "CV RMSE": result["cv_rmse_mean"],
        })

        if m["RMSE"] < best_rmse:
            best_rmse = m["RMSE"]
            best_name = name

        print(f"[evaluate] {name:22s}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}")

    df_res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    plot_model_comparison(rows, eda_dir)

    report = os.path.join(reports_dir, "final_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("# House Price Prediction — Final Report\n\n")
        f.write(f"**Best Model:** `{best_name}` | **Test RMSE:** `{best_rmse:.4f}`\n\n")
        f.write("## Test Set Results\n\n")
        f.write(df_res.to_markdown(index=False))
        f.write("\n\n## Notes\n")
        f.write("- Target: `log1p(MedHouseVal)` — exponentiate predictions with `expm1()`.\n")
        f.write("- Scaler saved to `models/scaler.joblib`.\n")
        f.write("- All plots saved to `reports/eda/`.\n")

    print(f"\n[evaluate] Best: {best_name} (RMSE={best_rmse:.4f})")
    print(f"[evaluate] Report -> {report}")
    return df_res
