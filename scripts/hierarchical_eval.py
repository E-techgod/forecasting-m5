"""Compute the official M5 metric: WRMSSE averaged across all 12 hierarchical levels."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting.data import PROCESSED_DIR, load_processed
from forecasting.evaluate import dollar_weights
from forecasting.hierarchy import LEVELS, build_meta, wrmsse

HORIZON = 28
FORECASTS_DIR = PROCESSED_DIR / "forecasts"


def main() -> None:
    print("Loading processed long table...")
    long = load_processed()
    meta = build_meta(long)

    print("Pivoting to wide (series x date)...")
    wide = long.pivot(index="id", columns="date", values="sales").astype("float32")
    wide = wide.sort_index(axis=1)

    dates = wide.columns
    cutoff_date = dates[-HORIZON]
    train_wide = wide.loc[:, dates < cutoff_date]
    actual_wide = wide.loc[:, dates >= cutoff_date]

    weights = dollar_weights(long, cutoff_date=cutoff_date, lookback_days=HORIZON)

    forecast_files = sorted(FORECASTS_DIR.glob("*.parquet"))
    if not forecast_files:
        raise FileNotFoundError(
            f"No forecast files in {FORECASTS_DIR}. Run run_baseline.py and train_lgbm.py first."
        )

    overall_rows = []
    level_rows = {}
    for path in forecast_files:
        model_name = path.stem
        forecast_wide = pd.read_parquet(path)
        forecast_wide = forecast_wide.reindex(index=actual_wide.index, columns=actual_wide.columns)
        forecast_wide = forecast_wide.fillna(0.0)

        overall, per_level = wrmsse(train_wide, actual_wide, forecast_wide, weights, meta)
        overall_rows.append({"model": model_name, "wrmsse": overall})
        level_rows[model_name] = per_level
        print(f"{model_name:20s}  WRMSSE={overall:.4f}")

    overall_df = pd.DataFrame(overall_rows).sort_values("wrmsse")
    print("\nOverall WRMSSE (mean across all 12 levels):")
    print(overall_df.to_string(index=False))
    overall_df.to_csv(PROCESSED_DIR / "wrmsse_results.csv", index=False)

    level_df = pd.DataFrame(level_rows).reindex([name for name, _ in LEVELS])
    print("\nPer-level RMSSE breakdown:")
    print(level_df.to_string())
    level_df.to_csv(PROCESSED_DIR / "wrmsse_by_level.csv")

    print(f"\nSaved {PROCESSED_DIR / 'wrmsse_results.csv'} and "
          f"{PROCESSED_DIR / 'wrmsse_by_level.csv'}")


if __name__ == "__main__":
    main()
