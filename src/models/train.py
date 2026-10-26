"""Model training — trains all models with cross-validation and saves artifacts."""
import os
import time
import yaml
import numpy as np
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_models(config: dict) -> dict:
    rs  = config["random_state"]
    mc  = config["models"]
    return {
        "baseline": DummyRegressor(strategy="median"),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=mc["ridge"]["alpha"]),
        "lasso": Lasso(alpha=mc["lasso"]["alpha"], max_iter=5000),
        "random_forest": RandomForestRegressor(
            n_estimators=mc["random_forest"]["n_estimators"],
            max_depth=mc["random_forest"]["max_depth"],
            random_state=rs, n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=mc["xgboost"]["n_estimators"],
            learning_rate=mc["xgboost"]["learning_rate"],
            max_depth=mc["xgboost"]["max_depth"],
            random_state=rs, verbosity=0,
        ),
    }


def train(X_train, y_train, config_path: str = "config.yaml") -> dict:
    """Train all models with KFold CV, serialize, and log to models/registry.md."""
    config     = load_config(config_path)
    models_dir = config["paths"]["models_dir"]
    cv_folds   = config["cv_folds"]
    os.makedirs(models_dir, exist_ok=True)

    models  = get_models(config)
    results = {}
    header  = "| Model | CV RMSE | CV Std | Train Time |\n|-------|---------|--------|------------|\n"
    rows    = []

    for name, model in models.items():
        print(f"[train] {name}...")
        t0 = time.time()
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        cv_mean = float(-scores.mean())
        cv_std  = float(scores.std())
        results[name] = {
            "model": model,
            "cv_rmse_mean": cv_mean,
            "cv_rmse_std":  cv_std,
            "train_time":   elapsed,
        }
        joblib.dump(model, os.path.join(models_dir, f"{name}.joblib"))
        rows.append(f"| {name} | {cv_mean:.4f} | {cv_std:.4f} | {elapsed:.1f}s |")
        print(f"  CV RMSE: {cv_mean:.4f} +/- {cv_std:.4f}  ({elapsed:.1f}s)")

    with open(os.path.join(models_dir, "registry.md"), "w", encoding="utf-8") as f:
        f.write("# Model Registry\n\n" + header + "\n".join(rows) + "\n")

    return results


if __name__ == "__main__":
    from src.features.engineer import engineer
    X_tr, X_te, y_tr, y_te, _ = engineer()
    train(X_tr, y_tr)
