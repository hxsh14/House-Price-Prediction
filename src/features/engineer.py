"""Feature engineering — cleans data, engineers features, scales, and splits."""
import os
import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_eda(df: pd.DataFrame, eda_dir: str) -> None:
    """Generate and save EDA plots."""
    os.makedirs(eda_dir, exist_ok=True)

    # 1. Target distribution (raw vs log-transformed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["MedHouseVal"], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Target Distribution (Raw)")
    axes[0].set_xlabel("MedHouseVal")
    axes[1].hist(np.log1p(df["MedHouseVal"]), bins=50, color="coral", edgecolor="white")
    axes[1].set_title("Target Distribution (Log-Transformed)")
    axes[1].set_xlabel("log1p(MedHouseVal)")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "target_distribution.png"), dpi=100)
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "correlation_heatmap.png"), dpi=100)
    plt.close()

    print(f"[engineer] EDA plots saved to {eda_dir}")


def engineer(config_path: str = "config.yaml"):
    """
    Load raw data, apply feature engineering, scale, split, and save artifacts.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    config = load_config(config_path)
    raw_path    = config["paths"]["raw_data"]
    proc_path   = config["paths"]["processed_data"]
    models_dir  = config["paths"]["models_dir"]
    eda_dir     = config["paths"]["eda_dir"]
    random_state = config["random_state"]
    test_size    = config["test_size"]

    df = pd.read_csv(raw_path)
    df = df.drop_duplicates().dropna()

    # EDA plots on raw data
    plot_eda(df, eda_dir)

    # Engineered features
    df["RoomsPerHousehold"] = df["AveRooms"] / df["HouseAge"].replace(0, np.nan)
    df["RoomsPerHousehold"] = df["RoomsPerHousehold"].fillna(df["RoomsPerHousehold"].median())

    df["PopPerHousehold"] = df["Population"] / df["AveOccup"].replace(0, np.nan)
    df["PopPerHousehold"] = df["PopPerHousehold"].fillna(df["PopPerHousehold"].median())

    # Log-transform target
    df["LogTarget"] = np.log1p(df["MedHouseVal"])

    target   = "LogTarget"
    features = [c for c in df.columns if c not in ("MedHouseVal", "LogTarget")]

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    pd.DataFrame(X_train, columns=features).assign(LogTarget=y_train).to_csv(proc_path, index=False)

    print(f"[engineer] Train={X_train.shape}  Test={X_test.shape}  Features={features}")
    return X_train, X_test, y_train, y_test, features


if __name__ == "__main__":
    engineer()
