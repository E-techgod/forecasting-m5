"""Train the global LightGBM model, evaluate on the same 28-day holdout as the baselines."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting.data import PROCESSED_DIR, load_processed
from forecasting.evaluate import dollar_weights, rmsse, summarize
from forecasting.features import CATEGORICAL_COLS, HORIZON, feature_columns
from forecasting.models import lgbm

HORIZON_DAYS = HORIZON  # 28, must match the baselines' holdout


def main() -> None:
    print("Loading feature table...")
    df = pd.read_parquet(PROCESSED_DIR / "m5_features.parquet")
    feature_cols = feature_columns(df)

    dates = sorted(df["date"].unique())
    test_cutoff = dates[-HORIZON_DAYS]
    valid_cutoff = dates[-2 * HORIZON_DAYS]

    # Drop rows without a full year of lookback (lag_364 is the binding constraint).
    usable = df.dropna(subset=["lag_364"])

    train_df = usable[usable["date"] < valid_cutoff]
    valid_df = usable[(usable["date"] >= valid_cutoff) & (usable["date"] < test_cutoff)]
    test_df = usable[usable["date"] >= test_cutoff]

    print(
        f"Train rows: {len(train_df):,} | Valid rows: {len(valid_df):,} | "
        f"Test rows: {len(test_df):,}"
    )

    print("Training LightGBM (tweedie objective)...")
    model = lgbm.train(train_df, valid_df, feature_cols, CATEGORICAL_COLS)

    print("Predicting on test horizon...")
    test_df = test_df.copy()
    test_df["prediction"] = lgbm.predict(model, test_df, feature_cols)

    # Reshape into the same wide (id x date) format the baseline evaluator expects.
    long = load_processed()
    actual_wide = long.pivot(index="id", columns="date", values="sales").astype("float32")
    actual_wide = actual_wide.sort_index(axis=1)
    train_wide = actual_wide.loc[:, actual_wide.columns < test_cutoff]
    actual_wide = actual_wide.loc[:, actual_wide.columns >= test_cutoff]

    forecast_wide = test_df.pivot(index="id", columns="date", values="prediction")
    # Some series may be missing rows if dropna trimmed them; align + fill with 0 as fallback.
    forecast_wide = forecast_wide.reindex(index=actual_wide.index, columns=actual_wide.columns)
    forecast_wide = forecast_wide.fillna(0.0)

    forecasts_dir = PROCESSED_DIR / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    forecast_wide.to_parquet(forecasts_dir / "lgbm_global.parquet")

    weights = dollar_weights(long, cutoff_date=test_cutoff, lookback_days=HORIZON_DAYS)
    scores = rmsse(train_wide, actual_wide, forecast_wide)
    summary = summarize(scores, weights)
    print(f"\nlgbm_global          mean_rmsse={summary['mean_rmsse']:.4f}  "
          f"weighted_rmsse={summary.get('weighted_rmsse', float('nan')):.4f}")

    per_series_path = PROCESSED_DIR / "per_series_rmsse.parquet"
    if per_series_path.exists():
        per_series = pd.read_parquet(per_series_path)
        per_series["lgbm_global"] = scores.reindex(per_series.index)
        per_series.to_parquet(per_series_path)
        print(f"Updated per-series scores at {per_series_path}")

    results_path = PROCESSED_DIR / "baseline_results.csv"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    results = results[results["model"] != "lgbm_global"]
    results = pd.concat(
        [results, pd.DataFrame([{"model": "lgbm_global", **summary}])], ignore_index=True
    )
    results = results.sort_values("weighted_rmsse")
    results.to_csv(results_path, index=False)
    print(f"\nUpdated comparison table at {results_path}")
    print(results.to_string(index=False))

    importance = pd.Series(
        model.feature_importance(importance_type="gain"), index=feature_cols
    ).sort_values(ascending=False)
    print("\nTop 10 features by gain:")
    print(importance.head(10).to_string())


if __name__ == "__main__":
    main()
