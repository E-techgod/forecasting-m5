"""Break down model performance by demand intermittency (fraction of zero-sales days).

The aggregate weighted RMSSE hides whether the LightGBM model's gain comes from fast
movers or actually generalizes to the sparse long tail — this checks that directly, and
tells us whether TSB is worth keeping as a per-segment fallback.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
BINS = [0.0, 0.2, 0.5, 0.8, 1.01]
LABELS = ["dense (<20% zero)", "moderate (20-50%)", "sparse (50-80%)", "very sparse (>80%)"]


def main() -> None:
    path = PROCESSED_DIR / "per_series_rmsse.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run run_baseline.py and train_lgbm.py first.")

    df = pd.read_parquet(path)
    model_cols = [c for c in df.columns if c not in ("zero_frac", "dollar_weight")]

    df["segment"] = pd.cut(df["zero_frac"], bins=BINS, labels=LABELS, right=False)

    print("Series count per segment:")
    print(df["segment"].value_counts().reindex(LABELS))

    print("\nMean RMSSE by segment (lower is better; unweighted across series in segment):")
    segment_means = df.groupby("segment", observed=True)[model_cols].mean()
    print(segment_means.reindex(LABELS).to_string())

    print("\nDollar-weighted RMSSE by segment:")

    def weighted_mean(g: pd.DataFrame) -> pd.Series:
        w = g["dollar_weight"]
        if w.sum() == 0:
            return g[model_cols].mean()
        return g[model_cols].apply(lambda col: (col * w).sum() / w.sum())

    segment_weighted = df.groupby("segment", observed=True).apply(
        weighted_mean, include_groups=False
    )
    print(segment_weighted.reindex(LABELS).to_string())

    best_per_segment = segment_means.reindex(LABELS).idxmin(axis=1)
    print("\nBest model per segment (unweighted mean RMSSE):")
    print(best_per_segment.to_string())

    out_path = PROCESSED_DIR / "segment_analysis.csv"
    segment_means.reindex(LABELS).to_csv(out_path)
    print(f"\nSaved segment breakdown to {out_path}")


if __name__ == "__main__":
    main()
