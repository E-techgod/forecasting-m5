"""Quantile forecasts for 28-day cumulative demand + a newsvendor inventory-cost simulation.

Closes the loop back to the original framing: a forecast is only useful insofar as it
drives a better reorder decision. We compare ordering to the point forecast's mean vs.
ordering to a quantile forecast at the newsvendor-optimal service level, and price both
in dollars (stockout cost vs. holding cost) instead of a forecast-error metric.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting.data import PROCESSED_DIR, load_processed
from forecasting.features import CATEGORICAL_COLS, HORIZON, add_target_28d, feature_columns
from forecasting.inventory import critical_ratio, pinball_loss, simulate_costs
from forecasting.models import lgbm

# Newsvendor cost assumptions, as a fraction of unit sell price:
CU_FRAC = 0.30  # underage: lost margin + goodwill cost per unit of unmet demand
CO_FRAC = 0.05  # overage: holding/markdown cost per unit of leftover inventory


def main() -> None:
    print("Loading feature table + processed long table...")
    features_df = pd.read_parquet(PROCESSED_DIR / "m5_features.parquet")
    long = load_processed()

    wide_sales = long.pivot(index="id", columns="date", values="sales").astype("float32")
    wide_sales = wide_sales.sort_index(axis=1)

    print("Computing 28-day-ahead cumulative demand target...")
    features_df = add_target_28d(features_df, wide_sales, horizon=HORIZON)

    dates = wide_sales.columns
    test_cutoff = dates[-HORIZON]
    valid_cutoff = dates[-2 * HORIZON]

    usable = features_df.dropna(subset=["lag_364", "target_28d"])
    train_df = usable[usable["date"] < valid_cutoff]
    valid_df = usable[(usable["date"] >= valid_cutoff) & (usable["date"] < test_cutoff)]
    eval_df = usable[usable["date"] == test_cutoff].set_index("id")

    feature_cols = feature_columns(usable)
    print(f"Train rows: {len(train_df):,} | Valid rows: {len(valid_df):,} | "
          f"Eval series: {len(eval_df):,}")

    ratio = critical_ratio(CU_FRAC, CO_FRAC)
    quantiles = sorted({0.5, 0.7, 0.9, 0.95, round(ratio, 3)})
    print(f"\nNewsvendor critical ratio (cu={CU_FRAC}, co={CO_FRAC}): {ratio:.3f}")
    print(f"Training quantile models for: {quantiles}\n")

    predictions = {}
    for q in quantiles:
        print(f"--- quantile {q} ---")
        model = lgbm.train_quantile(train_df, valid_df, feature_cols, CATEGORICAL_COLS, quantile=q)
        preds = model.predict(eval_df[feature_cols], num_iteration=model.best_iteration).clip(min=0)
        predictions[q] = pd.Series(preds, index=eval_df.index)

    actual_demand = eval_df["target_28d"]

    print("\nCalibration + pinball loss per quantile:")
    for q in quantiles:
        coverage = (actual_demand <= predictions[q]).mean()
        loss = pinball_loss(actual_demand, predictions[q], q)
        print(f"  q={q:.3f}  nominal_coverage={q:.3f}  actual_coverage={coverage:.3f}  "
              f"pinball_loss={loss:.4f}")

    # Order-quantity policies to compare.
    forecasts_dir = PROCESSED_DIR / "forecasts"
    ma_point = pd.read_parquet(forecasts_dir / "moving_average_28.parquet").sum(axis=1)
    lgbm_point = pd.read_parquet(forecasts_dir / "lgbm_global.parquet").sum(axis=1)

    nearest_q = min(quantiles, key=lambda q: abs(q - ratio))
    policies = {
        "point_moving_average": ma_point,
        "point_lgbm": lgbm_point,
        "quantile_median (q=0.5)": predictions[0.5],
        f"quantile_newsvendor (q={nearest_q})": predictions[nearest_q],
    }

    unit_price = (
        long[(long["date"] >= test_cutoff)].groupby("id")["sell_price"].mean()
    )

    print(f"\nInventory cost simulation (cu_frac={CU_FRAC}, co_frac={CO_FRAC}), "
          f"summed over {len(actual_demand):,} series:\n")
    rows = []
    for name, order_qty in policies.items():
        result = simulate_costs(order_qty, actual_demand, unit_price, CU_FRAC, CO_FRAC)
        rows.append({"policy": name, **result})
        print(f"  {name:32s}  total_cost=${result['total_cost']:,.0f}  "
              f"stockout=${result['stockout_cost']:,.0f}  "
              f"holding=${result['holding_cost']:,.0f}  "
              f"fill_rate={result['fill_rate']:.3f}")

    results = pd.DataFrame(rows).sort_values("total_cost")
    baseline_cost = results.loc[results["policy"] == "point_moving_average", "total_cost"].iloc[0]
    results["pct_savings_vs_moving_average"] = 1 - results["total_cost"] / baseline_cost

    out_path = PROCESSED_DIR / "inventory_simulation.csv"
    results.to_csv(out_path, index=False)
    print(f"\n{results.to_string(index=False)}")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
