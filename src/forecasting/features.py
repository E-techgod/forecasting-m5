"""Feature engineering for the global LightGBM model: lags, rolling stats, calendar, price."""

import numpy as np
import pandas as pd

HORIZON = 28  # M5's forecast horizon: we train a single model to predict 28 days ahead

# All lag/rolling features are anchored >= HORIZON days before the target date. This is
# deliberate: a model predicting day t+28 in one shot must not use any feature that falls
# inside the forecast window itself (e.g. a naive lag_7 for the last week of the horizon
# would secretly encode actual sales from earlier in that same unobserved window). Basing
# every lag on shift(28 + k) keeps the model honest about what's actually known 28 days out.
LAGS = (28, 35, 42, 56, 364)
ROLLING_WINDOWS = (7, 28, 90)
CATEGORICAL_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday"]


def add_lag_features(df: pd.DataFrame, lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    """Shifted sales values, computed per series. Must be called on data sorted by (id, date).

    `lags` are absolute shift amounts (already >= HORIZON), not offsets from the horizon.
    """
    g = df.groupby("id", sort=False)["sales"]
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag).astype("float32")
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Rolling mean/std of sales, anchored `horizon` days back so nothing leaks into the window."""
    shifted = df.groupby("id", sort=False)["sales"].shift(horizon)
    grouped_shifted = shifted.groupby(df["id"], sort=False)
    for w in windows:
        df[f"roll_mean_{w}"] = (
            grouped_shifted.rolling(w).mean().reset_index(level=0, drop=True).astype("float32")
        )
        df[f"roll_std_{w}"] = (
            grouped_shifted.rolling(w).std().reset_index(level=0, drop=True).astype("float32")
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["weekday"] = df["date"].dt.dayofweek.astype("int16")
    df["month"] = df["date"].dt.month.astype("int16")
    df["is_weekend"] = (df["weekday"] >= 5).astype("int8")
    df["is_event"] = df["event_name_1"].notna().astype("int8")

    snap_lookup = df[["snap_CA", "snap_TX", "snap_WI"]].to_numpy()
    state_idx = df["state_id"].map({"CA": 0, "TX": 1, "WI": 2}).to_numpy()
    df["snap"] = snap_lookup[np.arange(len(df)), state_idx].astype("int8")
    df = df.drop(columns=["snap_CA", "snap_TX", "snap_WI", "event_name_1", "event_type_1"])
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    item_mean_price = df.groupby("item_id", sort=False)["sell_price"].transform("mean")
    df["price_rel_to_item_mean"] = (df["sell_price"] / item_mean_price).astype("float32")
    return df


def build_feature_table(long: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline: sort, then lag/rolling/calendar/price features."""
    df = long.sort_values(["id", "date"]).reset_index(drop=True)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_price_features(df)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"id", "d", "date", "sales", "wm_yr_wk", "target_28d"}
    return [c for c in df.columns if c not in exclude]


def forward_sum_target(wide_sales: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """For each (series, date=t), the total demand over [t, t+horizon-1] — the cumulative
    demand a reorder decision made around `t` actually needs to cover. NaN past the point
    where the forward window runs off the end of the data.
    """
    cs = wide_sales.to_numpy(dtype="float64").cumsum(axis=1)
    cs_padded = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)
    n = wide_sales.shape[1]
    forward = np.full((wide_sales.shape[0], n), np.nan)
    for t in range(n - horizon + 1):
        forward[:, t] = cs_padded[:, t + horizon] - cs_padded[:, t]
    return pd.DataFrame(forward, index=wide_sales.index, columns=wide_sales.columns)


def add_target_28d(df: pd.DataFrame, wide_sales: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Merge the forward-sum target (long format) into a feature table on (id, date)."""
    target_wide = forward_sum_target(wide_sales, horizon)
    target_long = target_wide.stack(future_stack=True).rename("target_28d").reset_index()
    target_long.columns = ["id", "date", "target_28d"]
    return df.merge(target_long, on=["id", "date"], how="left")
