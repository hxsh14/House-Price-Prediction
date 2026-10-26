"""
main.py — House Price Prediction full pipeline entrypoint.

Usage:
    python main.py
"""
from src.data.ingest import ingest
from src.features.engineer import engineer
from src.models.train import train
from src.evaluate.metrics import evaluate

BANNER = "=" * 60


def main() -> None:
    print(f"\n{BANNER}")
    print("  House Price Prediction Pipeline")
    print(BANNER)

    print("\n[Phase 1 / D] Discovering data...")
    ingest()

    print("\n[Phase 2 / A] Analyzing & engineering features...")
    X_train, X_test, y_train, y_test, feature_names = engineer()

    print("\n[Phase 3 / T] Training models...")
    train_results = train(X_train, y_train)

    print("\n[Phase 4 / A] Assessing model performance...")
    evaluate(train_results, X_test, y_test, feature_names)

    print(f"\n{BANNER}")
    print("  Pipeline complete!")
    print("  -> Reports : reports/final_report.md")
    print("  -> Models  : models/")
    print("  -> Plots   : reports/eda/")
    print(BANNER + "\n")


if __name__ == "__main__":
    main()
