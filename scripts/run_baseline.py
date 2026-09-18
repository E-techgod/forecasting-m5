"""Fit baseline forecasters on M5, evaluate with RMSSE over the last 28 days, and compare."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting.baselines import BASELINES
from forecasting.data import PROCESSED_DIR, load_processed
from forecasting.evaluate import dollar_weights, rmsse, summarize, zero_fraction

HORIZON = 28  # M5's standard forecast horizon
FORECASTS_DIR = PROCESSED_DIR / "forecasts"


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

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    per_series = pd.DataFrame(index=train_wide.index)
    for name, forecast_fn in BASELINES.items():
        forecast_wide = forecast_fn(train_wide, HORIZON)
        forecast_wide.columns = actual_wide.columns
        forecast_wide.to_parquet(FORECASTS_DIR / f"{name}.parquet")
        scores = rmsse(train_wide, actual_wide, forecast_wide)
        per_series[name] = scores
        summary = summarize(scores, weights)
        rows.append({"model": name, **summary})
        print(f"{name:20s}  mean_rmsse={summary['mean_rmsse']:.4f}  "
              f"weighted_rmsse={summary.get('weighted_rmsse', float('nan')):.4f}")

    per_series["zero_frac"] = zero_fraction(train_wide)
    per_series["dollar_weight"] = weights.reindex(per_series.index).fillna(0)
    per_series_path = PROCESSED_DIR / "per_series_rmsse.parquet"
    per_series.to_parquet(per_series_path)
    print(f"Saved per-series scores to {per_series_path}")

    results = pd.DataFrame(rows).sort_values("weighted_rmsse")
    out_path = PROCESSED_DIR / "baseline_results.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved comparison table to {out_path}")


if __name__ == "__main__":
    main()
