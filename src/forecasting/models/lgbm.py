"""A single global LightGBM model trained across all M5 series, predicting 28 days ahead."""

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "objective": "tweedie",  # handles the zero-inflated, right-skewed sales distribution
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}


def train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        train_df[feature_cols],
        label=train_df["sales"],
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid_df[feature_cols],
        label=valid_df["sales"],
        categorical_feature=categorical_cols,
        reference=train_set,
        free_raw_data=False,
    )

    model = lgb.train(
        {**DEFAULT_PARAMS, **(params or {})},
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(period=50),
        ],
    )
    return model


def predict(model: lgb.Booster, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    preds = model.predict(df[feature_cols], num_iteration=model.best_iteration)
    return np.clip(preds, 0, None)
