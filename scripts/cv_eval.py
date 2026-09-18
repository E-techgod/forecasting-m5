"""Rolling-origin backtest: repeat the full evaluation on 3 non-overlapping 28-day windows
covering the last 84 days, to check the WRMSSE ranking isn't a fluke of one holdout period
(M5 itself had a public/private leaderboard shakeup from over-fitting to a single window).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting.baselines import BASELINES
from forecasting.data import PROCESSED_DIR, load_processed
from forecasting.evaluate import dollar_weights, rmsse, summarize
from forecasting.features import CATEGORICAL_COLS, feature_columns
from forecasting.hierarchy import build_meta, wrmsse
from forecasting.models import lgbm

HORIZON = 28
OFFSETS = [0, 28, 56]  # 0 = most recent 28 days, 28 = the 28 days before that, etc.


def run_window(offset: int, wide: pd.DataFrame, features_df: pd.DataFrame, long: pd.DataFrame, meta) -> list[dict]:
    dates = wide.columns
    test_start_idx = len(dates) - HORIZON - offset
    test_cutoff = dates[test_start_idx]
    test_end = dates[test_start_idx + HORIZON - 1]

    train_wide = wide.loc[:, dates < test_cutoff]
    actual_wide = wide.loc[:, (dates >= test_cutoff) & (dates <= test_end)]
    weights = dollar_weights(long, cutoff_date=test_cutoff, lookback_days=HORIZON)

    rows = []
    forecasts = {}

    for name, forecast_fn in BASELINES.items():
        forecast_wide = forecast_fn(train_wide, HORIZON)
        forecast_wide.columns = actual_wide.columns
        forecasts[name] = forecast_wide

    valid_cutoff = dates[test_start_idx - HORIZON]
    usable = features_df.dropna(subset=["lag_364"])
    train_df = usable[usable["date"] < valid_cutoff]
    valid_df = usable[(usable["date"] >= valid_cutoff) & (usable["date"] < test_cutoff)]
    test_df = usable[(usable["date"] >= test_cutoff) & (usable["date"] <= test_end)].copy()

    feature_cols = feature_columns(usable)
    model = lgbm.train(train_df, valid_df, feature_cols, CATEGORICAL_COLS)
    test_df["prediction"] = lgbm.predict(model, test_df, feature_cols)
    lgbm_forecast = test_df.pivot(index="id", columns="date", values="prediction")
    lgbm_forecast = lgbm_forecast.reindex(index=actual_wide.index, columns=actual_wide.columns).fillna(0.0)
    forecasts["lgbm_global"] = lgbm_forecast

    for name, forecast_wide in forecasts.items():
        scores = rmsse(train_wide, actual_wide, forecast_wide)
        bottom_summary = summarize(scores, weights)
        overall, _ = wrmsse(train_wide, actual_wide, forecast_wide, weights, meta)
        rows.append(
            {
                "offset": offset,
                "model": name,
                "bottom_weighted_rmsse": bottom_summary.get("weighted_rmsse", float("nan")),
                "wrmsse": overall,
            }
        )
        print(f"  window offset={offset:>3}  {name:20s}  "
              f"bottom_wrmsse={bottom_summary.get('weighted_rmsse', float('nan')):.4f}  "
              f"full_wrmsse={overall:.4f}")

    return rows


def main() -> None:
    print("Loading processed long table + features...")
    long = load_processed()
    meta = build_meta(long)
    features_df = pd.read_parquet(PROCESSED_DIR / "m5_features.parquet")

    wide = long.pivot(index="id", columns="date", values="sales").astype("float32")
    wide = wide.sort_index(axis=1)

    all_rows = []
    for offset in OFFSETS:
        print(f"\n=== Window offset={offset} ===")
        all_rows.extend(run_window(offset, wide, features_df, long, meta))

    results = pd.DataFrame(all_rows)
    out_path = PROCESSED_DIR / "cv_results.csv"
    results.to_csv(out_path, index=False)

    print("\nStability across windows (mean +/- std of full WRMSSE):")
    stability = results.groupby("model")["wrmsse"].agg(["mean", "std"]).sort_values("mean")
    print(stability.to_string())
    stability.to_csv(PROCESSED_DIR / "cv_stability.csv")

    print(f"\nSaved {out_path} and {PROCESSED_DIR / 'cv_stability.csv'}")


if __name__ == "__main__":
    main()
