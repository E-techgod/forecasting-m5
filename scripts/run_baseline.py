"""Fit baseline forecasters on M5, evaluate with RMSSE over the last 28 days, and compare."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting.baselines import BASELINES
from forecasting.data import load_processed
from forecasting.evaluate import dollar_weights, rmsse, summarize

HORIZON = 28  # M5's standard forecast horizon


def main() -> None:
    print("Loading processed long table...")
    long = load_processed()

    print("Pivoting to wide (series x date)...")
    wide = long.pivot(index="id", columns="date", values="sales").astype("float32")
    wide = wide.sort_index(axis=1)

    dates = wide.columns
    cutoff_date = dates[-HORIZON]
    train_wide = wide.loc[:, dates < cutoff_date]
    actual_wide = wide.loc[:, dates >= cutoff_date]

    weights = dollar_weights(long, cutoff_date=cutoff_date, lookback_days=HORIZON)

    print(f"\nTrain: {train_wide.shape[1]} days, Test: {actual_wide.shape[1]} days, "
          f"Series: {train_wide.shape[0]:,}\n")

    rows = []
    for name, forecast_fn in BASELINES.items():
        forecast_wide = forecast_fn(train_wide, HORIZON)
        forecast_wide.columns = actual_wide.columns
        scores = rmsse(train_wide, actual_wide, forecast_wide)
        summary = summarize(scores, weights)
        rows.append({"model": name, **summary})
        print(f"{name:20s}  mean_rmsse={summary['mean_rmsse']:.4f}  "
              f"weighted_rmsse={summary.get('weighted_rmsse', float('nan')):.4f}")

    results = pd.DataFrame(rows).sort_values("weighted_rmsse")
    out_path = Path("data/processed/baseline_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\nSaved comparison table to {out_path}")


if __name__ == "__main__":
    main()
