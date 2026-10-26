"""Tests for src/features/engineer.py"""
import numpy as np
import pytest


def make_config(tmp_path):
    return {
        "random_state": 42,
        "test_size": 0.2,
        "paths": {
            "raw_data":        str(tmp_path / "housing.csv"),
            "processed_data":  str(tmp_path / "processed.csv"),
            "models_dir":      str(tmp_path / "models"),
            "eda_dir":         str(tmp_path / "eda"),
        },
    }


def run_pipeline(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    monkeypatch.setattr("src.data.ingest.load_config",       lambda *a, **k: cfg)
    monkeypatch.setattr("src.features.engineer.load_config", lambda *a, **k: cfg)
    from src.data.ingest import ingest
    from src.features.engineer import engineer
    ingest()
    return engineer()


def test_no_nans_in_X_train(tmp_path, monkeypatch):
    X_train, X_test, y_train, y_test, _ = run_pipeline(tmp_path, monkeypatch)
    assert not np.isnan(X_train).any()


def test_no_nans_in_X_test(tmp_path, monkeypatch):
    X_train, X_test, y_train, y_test, _ = run_pipeline(tmp_path, monkeypatch)
    assert not np.isnan(X_test).any()


def test_train_test_split_ratio(tmp_path, monkeypatch):
    X_train, X_test, y_train, y_test, _ = run_pipeline(tmp_path, monkeypatch)
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.2) < 0.02


def test_engineered_features_present(tmp_path, monkeypatch):
    X_train, X_test, y_train, y_test, features = run_pipeline(tmp_path, monkeypatch)
    assert "RoomsPerHousehold" in features
    assert "PopPerHousehold" in features
