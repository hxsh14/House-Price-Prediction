"""Tests for src/models/train.py"""
import numpy as np
import pytest


def make_config(tmp_path):
    return {
        "random_state": 42,
        "test_size": 0.2,
        "cv_folds": 2,
        "paths": {
            "raw_data":       str(tmp_path / "housing.csv"),
            "processed_data": str(tmp_path / "processed.csv"),
            "models_dir":     str(tmp_path / "models"),
            "eda_dir":        str(tmp_path / "eda"),
        },
        "models": {
            "ridge":         {"alpha": 1.0},
            "lasso":         {"alpha": 0.1},
            "random_forest": {"n_estimators": 10, "max_depth": 5},
            "xgboost":       {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
        },
    }


def run_pipeline(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    monkeypatch.setattr("src.data.ingest.load_config",       lambda *a, **k: cfg)
    monkeypatch.setattr("src.features.engineer.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("src.models.train.load_config",      lambda *a, **k: cfg)
    from src.data.ingest import ingest
    from src.features.engineer import engineer
    from src.models.train import train
    ingest()
    X_train, X_test, y_train, y_test, features = engineer()
    results = train(X_train, y_train)
    return results, X_test, y_test


def test_all_models_return_predictions(tmp_path, monkeypatch):
    results, X_test, y_test = run_pipeline(tmp_path, monkeypatch)
    for name, result in results.items():
        preds = result["model"].predict(X_test)
        assert len(preds) == len(y_test), f"{name}: prediction length mismatch"
        assert not np.isnan(preds).any(), f"{name}: NaN predictions"


def test_models_beat_baseline(tmp_path, monkeypatch):
    results, _, _ = run_pipeline(tmp_path, monkeypatch)
    baseline_rmse = results["baseline"]["cv_rmse_mean"]
    for name in ("ridge", "random_forest", "xgboost"):
        assert results[name]["cv_rmse_mean"] < baseline_rmse, \
            f"{name} CV RMSE should beat baseline"


def test_all_models_have_cv_rmse(tmp_path, monkeypatch):
    results, _, _ = run_pipeline(tmp_path, monkeypatch)
    for name, result in results.items():
        assert result["cv_rmse_mean"] > 0, f"{name}: invalid CV RMSE"
