"""Reshape raw M5 CSVs into a single long parquet table for modeling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting.data import build_long_table, load_raw, save_processed


def main() -> None:
    print("Loading raw CSVs...")
    sales, calendar, prices = load_raw()

    print("Melting + joining calendar/prices...")
    long = build_long_table(sales, calendar, prices)

    out_path = save_processed(long)
    print(f"Saved {len(long):,} rows to {out_path}")


if __name__ == "__main__":
    main()
