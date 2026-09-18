"""The 12 official M5 aggregation levels, and WRMSSE = mean RMSSE across all of them.

RMSSE of an aggregated series is NOT the aggregate of bottom-level RMSSEs (errors can
cancel out when series are summed), so each level's actual/forecast/train matrices are
rebuilt by summing the bottom-level (item x store) matrices within each grouping, and
RMSSE is computed fresh at that level.
"""

import pandas as pd

from forecasting.evaluate import rmsse

LEVELS: list[tuple[str, list[str]]] = [
    ("total", []),
    ("state", ["state_id"]),
    ("store", ["store_id"]),
    ("category", ["cat_id"]),
    ("department", ["dept_id"]),
    ("state_category", ["state_id", "cat_id"]),
    ("state_department", ["state_id", "dept_id"]),
    ("store_category", ["store_id", "cat_id"]),
    ("store_department", ["store_id", "dept_id"]),
    ("item", ["item_id"]),
    ("item_state", ["item_id", "state_id"]),
    ("item_store", ["item_id", "store_id"]),  # bottom level, same partition as the raw `id`
]


def build_meta(long: pd.DataFrame) -> pd.DataFrame:
    """One row per bottom-level series id, with its grouping columns."""
    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    meta = long[cols].drop_duplicates().set_index("id")
    return meta


def _group_keys(meta: pd.DataFrame, index: pd.Index, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series("total", index=index)
    aligned = meta.loc[index, cols].astype(str)
    if len(cols) == 1:
        return aligned[cols[0]]
    return aligned.agg("|".join, axis=1)


def aggregate_wide(wide: pd.DataFrame, meta: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keys = _group_keys(meta, wide.index, cols)
    return wide.groupby(keys).sum()


def aggregate_weights(weights: pd.Series, meta: pd.DataFrame, cols: list[str]) -> pd.Series:
    keys = _group_keys(meta, weights.index, cols)
    return weights.groupby(keys).sum()


def level_wrmsse(
    train_wide: pd.DataFrame,
    actual_wide: pd.DataFrame,
    forecast_wide: pd.DataFrame,
    weights: pd.Series,
    meta: pd.DataFrame,
    cols: list[str],
) -> float:
    train_agg = aggregate_wide(train_wide, meta, cols)
    actual_agg = aggregate_wide(actual_wide, meta, cols)
    forecast_agg = aggregate_wide(forecast_wide, meta, cols)
    weights_agg = aggregate_weights(weights, meta, cols)

    scores = rmsse(train_agg, actual_agg, forecast_agg)
    clean = scores.dropna()
    aligned_weights = weights_agg.reindex(clean.index).fillna(0)
    if aligned_weights.sum() == 0:
        return float(clean.mean())
    return float((clean * aligned_weights).sum() / aligned_weights.sum())


def wrmsse(
    train_wide: pd.DataFrame,
    actual_wide: pd.DataFrame,
    forecast_wide: pd.DataFrame,
    weights: pd.Series,
    meta: pd.DataFrame,
) -> tuple[float, pd.Series]:
    """Returns (overall WRMSSE, per-level WRMSSE breakdown)."""
    per_level = {}
    for name, cols in LEVELS:
        per_level[name] = level_wrmsse(train_wide, actual_wide, forecast_wide, weights, meta, cols)
    per_level_series = pd.Series(per_level)
    return float(per_level_series.mean()), per_level_series
