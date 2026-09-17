"""Simple per-series baseline forecasters, operating on a wide (series x date) matrix."""

import numpy as np
import pandas as pd


def naive_forecast(train_wide: pd.DataFrame, h: int) -> pd.DataFrame:
    """Repeat the last observed value for all h future steps."""
    last = train_wide.iloc[:, -1]
    return pd.DataFrame(
        np.tile(last.to_numpy()[:, None], (1, h)), index=train_wide.index
    )


def seasonal_naive_forecast(
    train_wide: pd.DataFrame, h: int, season_length: int = 7
) -> pd.DataFrame:
    """Repeat the last full season (default: last 7 days) cyclically."""
    last_season = train_wide.iloc[:, -season_length:].to_numpy()
    reps = int(np.ceil(h / season_length))
    tiled = np.tile(last_season, (1, reps))[:, :h]
    return pd.DataFrame(tiled, index=train_wide.index)


def moving_average_forecast(
    train_wide: pd.DataFrame, h: int, window: int = 28
) -> pd.DataFrame:
    """Repeat the mean of the last `window` days for all h future steps."""
    avg = train_wide.iloc[:, -window:].mean(axis=1)
    return pd.DataFrame(
        np.tile(avg.to_numpy()[:, None], (1, h)), index=train_wide.index
    )


BASELINES = {
    "naive": naive_forecast,
    "seasonal_naive_7": lambda tw, h: seasonal_naive_forecast(tw, h, season_length=7),
    "moving_average_28": lambda tw, h: moving_average_forecast(tw, h, window=28),
}
