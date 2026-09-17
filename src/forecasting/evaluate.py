"""M5-style evaluation: RMSSE per series, and a dollar-sales-weighted average (WRMSSE-lite).

Note: this weights only at the bottom (series) level using last-28-days dollar sales as
a proxy for the official M5 weight. The real competition metric (WRMSSE) also averages
across 12 hierarchical aggregation levels (total, state, category, store-item, ...) —
that's a natural follow-up once the baseline numbers look sane.
"""

import numpy as np
import pandas as pd


def rmsse(
    train_wide: pd.DataFrame, actual_wide: pd.DataFrame, forecast_wide: pd.DataFrame
) -> pd.Series:
    """Root Mean Squared Scaled Error, scaled by in-sample naive one-step-ahead error."""
    train = train_wide.to_numpy(dtype=float)
    actual = actual_wide.to_numpy(dtype=float)
    forecast = forecast_wide.to_numpy(dtype=float)

    n = train.shape[1]
    naive_sq_diff = np.square(np.diff(train, axis=1))
    scale = naive_sq_diff.sum(axis=1) / (n - 1)

    h = actual.shape[1]
    forecast_sq_err = np.square(actual - forecast).sum(axis=1) / h

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.sqrt(forecast_sq_err / scale)
    result = np.where(scale == 0, np.nan, result)
    return pd.Series(result, index=train_wide.index, name="rmsse")


def dollar_weights(long: pd.DataFrame, cutoff_date: pd.Timestamp, lookback_days: int = 28) -> pd.Series:
    """Approximate M5 series weights: total $ sales in the last `lookback_days` before cutoff."""
    window_start = cutoff_date - pd.Timedelta(days=lookback_days)
    recent = long[(long["date"] >= window_start) & (long["date"] < cutoff_date)].copy()
    recent["dollar_sales"] = recent["sales"] * recent["sell_price"].fillna(0)
    weights = recent.groupby("id")["dollar_sales"].sum()
    total = weights.sum()
    return weights / total if total > 0 else weights


def summarize(scores: pd.Series, weights: pd.Series | None = None) -> dict:
    clean = scores.dropna()
    out = {"mean_rmsse": clean.mean(), "median_rmsse": clean.median(), "n_series": len(clean)}
    if weights is not None:
        aligned = weights.reindex(clean.index).fillna(0)
        if aligned.sum() > 0:
            out["weighted_rmsse"] = float((clean * aligned).sum() / aligned.sum())
    return out
