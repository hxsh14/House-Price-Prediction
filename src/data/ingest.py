"""Data ingestion module — loads California Housing dataset and saves to data/raw/."""
import os
import pandas as pd
import yaml
from sklearn.datasets import fetch_california_housing


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ingest(config_path: str = "config.yaml") -> pd.DataFrame:
    """Fetch California Housing dataset and persist to data/raw/housing.csv."""
    config = load_config(config_path)
    raw_path = config["paths"]["raw_data"]

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # includes target column MedHouseVal

    df.to_csv(raw_path, index=False)
    print(f"[ingest] Saved {len(df)} rows -> {raw_path}")
    return df


if __name__ == "__main__":
    ingest()
