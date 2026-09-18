"""Newsvendor-style cost simulation: turn a demand forecast into an order quantity and
price the outcome against realized demand, so forecast accuracy is judged by the dollars
it saves/costs, not just an error metric.
"""

import numpy as np
import pandas as pd


def critical_ratio(cu: float, co: float) -> float:
    """Optimal service level for a single-period newsvendor problem.

    cu: underage cost per unit short (lost margin/goodwill from a stockout)
    co: overage cost per unit left over (holding/markdown cost)
    """
    return cu / (cu + co)


def simulate_costs(
    order_qty: pd.Series,
    actual_demand: pd.Series,
    unit_price: pd.Series,
    cu_frac: float,
    co_frac: float,
) -> dict:
    """Total cost of ordering `order_qty` against realized `actual_demand`, per series.

    Unit stockout cost = cu_frac * price, unit holding cost = co_frac * price — costs
    scale with price so a $1 miss on a $50 item isn't weighted the same as on a $2 item.
    """
    idx = order_qty.index
    demand = actual_demand.reindex(idx).fillna(0.0).to_numpy()
    order = order_qty.fillna(0.0).to_numpy()
    price = unit_price.reindex(idx).fillna(0.0).to_numpy()

    shortage = np.clip(demand - order, 0, None)
    excess = np.clip(order - demand, 0, None)

    stockout_cost = shortage * price * cu_frac
    holding_cost = excess * price * co_frac
    total_cost = stockout_cost + holding_cost

    return {
        "total_cost": float(total_cost.sum()),
        "stockout_cost": float(stockout_cost.sum()),
        "holding_cost": float(holding_cost.sum()),
        "n_series": len(idx),
        "fill_rate": float((demand <= order).mean()),
    }


def pinball_loss(actual: pd.Series, predicted_quantile: pd.Series, quantile: float) -> float:
    idx = actual.index
    y = actual.reindex(idx).to_numpy()
    yhat = predicted_quantile.reindex(idx).to_numpy()
    diff = y - yhat
    loss = np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)
    return float(np.nanmean(loss))
