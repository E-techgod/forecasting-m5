"""Load and reshape the M5 Forecasting dataset into a long (series, date) table."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

SALES_FILE = "sales_train_evaluation.csv"  # falls back to sales_train_validation.csv
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"


def _sales_path() -> Path:
    eval_path = RAW_DIR / SALES_FILE
    if eval_path.exists():
        return eval_path
    val_path = RAW_DIR / "sales_train_validation.csv"
    if val_path.exists():
        return val_path
    raise FileNotFoundError(
        f"No sales file found in {RAW_DIR}. Run `uv run scripts/download_m5.py` first."
    )


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw M5 CSVs as-is."""
    sales = pd.read_csv(_sales_path())
    calendar = pd.read_csv(RAW_DIR / CALENDAR_FILE, parse_dates=["date"])
    prices = pd.read_csv(RAW_DIR / PRICES_FILE)
    return sales, calendar, prices


def melt_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Wide (id, d_1..d_N) -> long (id, d, sales)."""
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(
        id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales"
    )
    return long


def _downcast(long: pd.DataFrame) -> pd.DataFrame:
    """Shrink dtypes: ~60M rows of plain strings/int64 would otherwise eat >10GB in memory."""
    category_cols = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "d", "event_name_1", "event_type_1",
    ]
    for col in category_cols:
        long[col] = long[col].astype("category")
    long["sales"] = long["sales"].astype("int16")
    long["wm_yr_wk"] = long["wm_yr_wk"].astype("int32")
    for col in ("snap_CA", "snap_TX", "snap_WI"):
        long[col] = long[col].astype("int8")
    long["sell_price"] = long["sell_price"].astype("float32")
    return long


def build_long_table(
    sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Join melted sales with calendar (date/events) and prices (weekly, per store-item)."""
    long = melt_sales(sales)
    long = long.merge(
        calendar[["d", "date", "wm_yr_wk", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]],
        on="d",
        how="left",
    )
    long = long.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )
    long = long.sort_values(["id", "date"]).reset_index(drop=True)
    long = _downcast(long)
    return long


def save_processed(long: pd.DataFrame, name: str = "m5_long.parquet") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / name
    long.to_parquet(out_path, index=False)
    return out_path


def load_processed(name: str = "m5_long.parquet") -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run scripts/build_dataset.py` first."
        )
    return pd.read_parquet(path)
