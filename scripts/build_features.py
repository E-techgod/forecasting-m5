"""Build the lag/rolling/calendar/price feature table and cache it as parquet."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting.data import PROCESSED_DIR, load_processed
from forecasting.features import build_feature_table


def main() -> None:
    print("Loading processed long table...")
    long = load_processed()

    print("Building features (lags, rolling stats, calendar, price)...")
    features = build_feature_table(long)

    out_path = PROCESSED_DIR / "m5_features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"Saved {len(features):,} rows x {features.shape[1]} cols to {out_path}")


if __name__ == "__main__":
    main()
