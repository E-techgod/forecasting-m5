# forecasting

SKU-level demand forecasting on the [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset (Walmart, 42,840 series, hierarchical, hourly-to-daily granularity, 28-day horizon).

## Setup

1. Get a Kaggle API token: https://www.kaggle.com/settings -> API -> "Create New Token" -> save `kaggle.json` to `~/.kaggle/kaggle.json` (`chmod 600`).
2. Join the competition (free, one click): https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules
3. Download + extract the raw CSVs:
   ```
   uv run scripts/download_m5.py
   ```
4. Reshape into a single long parquet table (melts wide day-columns, joins calendar + prices):
   ```
   uv run scripts/build_dataset.py
   ```
5. Run the baseline forecasters and compare:
   ```
   uv run scripts/run_baseline.py
   ```

## What's here

- `src/forecasting/data.py` — raw CSV loading + reshape to long format (`data/processed/m5_long.parquet`)
- `src/forecasting/baselines.py` — naive, seasonal-naive (7-day), moving-average (28-day) forecasters
- `src/forecasting/evaluate.py` — RMSSE per series (M5's official scaled error) + a dollar-sales-weighted average
- `scripts/` — CLI entry points for each stage of the pipeline

## Evaluation

`run_baseline.py` holds out the last 28 days as the test horizon (matching M5's competition horizon), fits each baseline on the remaining history, and scores with **RMSSE** (root mean squared error scaled by each series' in-sample naive one-step error — handles the wide range of series volumes without needing per-series normalization).

The `weighted_rmsse` column approximates the official **WRMSSE** metric by weighting each series by its trailing 28-day dollar sales. The full WRMSSE also averages across 12 hierarchical aggregation levels (total, state, category, store x item, ...) — not yet implemented here.

## Next steps

- Intermittent-demand baselines (Croston, TSB) for sparse long-tail SKUs
- A global LightGBM model with lag/rolling/calendar features, trained across all series at once
- Full hierarchical WRMSSE (all 12 M5 aggregation levels) + reconciliation (MinT)
- Quantile forecasts feeding a newsvendor reorder policy
