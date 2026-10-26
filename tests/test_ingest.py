"""Tests for src/data/ingest.py"""
import os
import pytest
import pandas as pd


def make_config(tmp_path):
    return {
        "paths": {"raw_data": str(tmp_path / "housing.csv")}
    }


def test_ingest_creates_csv(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    monkeypatch.setattr("src.data.ingest.load_config", lambda *a, **k: cfg)
    from src.data.ingest import ingest
    df = ingest()
    assert os.path.exists(cfg["paths"]["raw_data"])
    assert len(df) > 0


def test_ingest_has_target_column(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    monkeypatch.setattr("src.data.ingest.load_config", lambda *a, **k: cfg)
    from src.data.ingest import ingest
    df = ingest()
    assert "MedHouseVal" in df.columns


def test_ingest_no_all_null_columns(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    monkeypatch.setattr("src.data.ingest.load_config", lambda *a, **k: cfg)
    from src.data.ingest import ingest
    df = ingest()
    assert df.isnull().all(axis=0).sum() == 0


def test_ingest_minimum_rows(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    monkeypatch.setattr("src.data.ingest.load_config", lambda *a, **k: cfg)
    from src.data.ingest import ingest
    df = ingest()
    assert df.shape[0] > 1000
