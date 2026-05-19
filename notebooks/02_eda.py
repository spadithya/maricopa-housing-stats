"""
Phase 2 — Exploratory Data Analysis
====================================

 Before we touch a model, we need to understand the data:
 what does the target look like, what features actually relate to price,
 and what surprises are lurking.

This script answers six questions, in order:

  Q1. What does SALE_PRICE actually look like, and why do we need log?
  Q2. Has price drifted year-over-year? (Inflation control)
  Q3. Which continuous features matter? (sqft, year built)
  Q4. Which categorical features matter? (Class, AC type, BOOK)
  Q5. How much missingness do we have, and where?
  Q6. What outliers / weirdness should we watch out for?

Plots are saved to data/processed/plots/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
PLOTS = PROCESSED / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# Nicer default styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    path = PROCESSED / "residential_with_sales.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path.name}")
    print(f"Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Q1: Target distribution — raw and log
# ---------------------------------------------------------------------------
def q1_target_distribution(df: pd.DataFrame) -> None:
    """Show why we model log(price) instead of price."""
    print("\n=== Q1: TARGET DISTRIBUTION ===")
    p = df["SALE_PRICE"]
    print(p.describe().apply(lambda x: f"{x:,.0f}"))

    # Skewness and kurtosis — quantitative measures of "how non-normal"
    print(f"  skewness:  {p.skew():.2f}   (0 = symmetric, >1 = heavy right tail)")
    print(f"  kurtosis:  {p.kurt():.2f}   (>3 = heavy tails)")
    print(f"  log skew:  {np.log(p).skew():.2f}   (after log transform)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(p, bins=80, color="steelblue", edgecolor="white")
    axes[0].set_title("SALE_PRICE (raw)")
    axes[0].set_xlabel("$")
    axes[0].ticklabel_format(style="plain", axis="x")
    axes[0].axvline(p.median(), color="orange", linestyle="--", label=f"median ${p.median():,.0f}")
    axes[0].axvline(p.mean(),   color="red",    linestyle="--", label=f"mean   ${p.mean():,.0f}")
    axes[0].legend()

    axes[1].hist(np.log(p), bins=80, color="seagreen", edgecolor="white")
    axes[1].set_title("log(SALE_PRICE)")
    axes[1].set_xlabel("log $")

    plt.tight_layout()
    out = PLOTS / "q1_target_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


# ---------------------------------------------------------------------------
# Q2: Has price drifted year-over-year?
# ---------------------------------------------------------------------------
def q2_price_by_year(df: pd.DataFrame) -> None:
    print("\n=== Q2: PRICE BY YEAR ===")
    df = df.copy()
    df["sale_year"] = df["SALE_DATE"].dt.year
    by_year = df.groupby("sale_year")["SALE_PRICE"].agg(["count", "median", "mean"])
    print(by_year.to_string(float_format=lambda x: f"{x:,.0f}"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(by_year.index, by_year["median"], marker="o", label="median")
    ax.plot(by_year.index, by_year["mean"],   marker="s", label="mean")
    ax.set_title("Maricopa County sale price by year (filtered set)")
    ax.set_ylabel("$")
    ax.set_xlabel("sale year")
    ax.ticklabel_format(style="plain", axis="y")
    ax.legend()
    plt.tight_layout()
    out = PLOTS / "q2_price_by_year.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


# ---------------------------------------------------------------------------
# Q3: Continuous features vs price
# ---------------------------------------------------------------------------
def q3_continuous_features(df: pd.DataFrame) -> None:
    print("\n=== Q3: CONTINUOUS FEATURES vs PRICE ===")

    cont = ["Living_sqft", "ConstructionYear", "FirstFloor_sqft",
            "SecondFloor_sqft", "POOL_SQFT", "ADDED_SQFT", "DETACH_SQFT",
            "Basement_sqft", "BathroomFixtures", "StoryCount", "ProportionComplete"]

    # Correlation with log(price), since price is log-normal
    corr = (df[cont].apply(pd.to_numeric, errors="coerce")
                    .assign(log_price=np.log(df["SALE_PRICE"]))
                    .corr()["log_price"]
                    .drop("log_price")
                    .sort_values(ascending=False))
    print("Pearson r with log(SALE_PRICE):")
    print(corr.to_string(float_format=lambda x: f"{x:+.3f}"))

    # Scatter the two strongest continuous predictors
    top2 = corr.abs().sort_values(ascending=False).head(2).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, feat in zip(axes, top2):
        sample = df.sample(min(20_000, len(df)), random_state=0)
        ax.scatter(sample[feat], np.log(sample["SALE_PRICE"]),
                   s=2, alpha=0.15, color="steelblue")
        ax.set_xlabel(feat)
        ax.set_ylabel("log(SALE_PRICE)")
        ax.set_title(f"{feat}  (r = {corr[feat]:+.3f})")
    plt.tight_layout()
    out = PLOTS / "q3_continuous_vs_price.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


# ---------------------------------------------------------------------------
# Q4: Categorical features — which ones move price?
# ---------------------------------------------------------------------------
def q4_categorical_features(df: pd.DataFrame) -> None:
    print("\n=== Q4: CATEGORICAL FEATURES ===")
    cats = ["Class", "AirConditioningType", "HeatingType",
            "ExteriorWallMaterial", "RoofMaterial", "PUC", "BOOK"]

    for c in cats:
        n_unique = df[c].nunique(dropna=False)
        print(f"\n  {c}: {n_unique} unique values")
        if n_unique <= 12:
            tbl = (df.groupby(c)["SALE_PRICE"]
                     .agg(["count", "median"])
                     .sort_values("count", ascending=False))
            print(tbl.to_string(float_format=lambda x: f"{x:,.0f}"))

    # Top BOOKs by sample size, ranked by median price
    print("\n  Top 15 BOOKs (by count) with median price:")
    book_stats = (df.groupby("BOOK")["SALE_PRICE"]
                    .agg(["count", "median"])
                    .sort_values("count", ascending=False)
                    .head(15))
    print(book_stats.to_string(float_format=lambda x: f"{x:,.0f}"))


# ---------------------------------------------------------------------------
# Q5: Missingness
# ---------------------------------------------------------------------------
def q5_missingness(df: pd.DataFrame) -> None:
    print("\n=== Q5: MISSINGNESS ===")
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        print("  No missing values in the filtered set!")
    else:
        print((miss * 100).round(2).astype(str) + " %")


# ---------------------------------------------------------------------------
# Q6: Outliers and oddities to watch
# ---------------------------------------------------------------------------
def q6_outliers(df: pd.DataFrame) -> None:
    print("\n=== Q6: OUTLIERS / ODDITIES ===")

    # Tiny or implausible homes
    tiny = (df["Living_sqft"] < 300).sum()
    huge = (df["Living_sqft"] > 10_000).sum()
    print(f"  Living_sqft < 300:    {tiny:,}   (probably data errors)")
    print(f"  Living_sqft > 10,000: {huge:,}   (mansions or data errors?)")

    # Construction year edge cases
    very_old = (df["ConstructionYear"] < 1900).sum()
    future   = (df["ConstructionYear"] > 2026).sum()
    print(f"  ConstructionYear < 1900: {very_old:,}")
    print(f"  ConstructionYear > 2026: {future:,}")

    # Duplicate parcels (same APN sold multiple times)
    n_parcels = df["PARCELNUMBER"].nunique()
    print(f"  Distinct parcels: {n_parcels:,} out of {len(df):,} rows")
    print(f"  → about {len(df) / n_parcels:.2f} sales per parcel on average")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load()
    q1_target_distribution(df)
    q2_price_by_year(df)
    q3_continuous_features(df)
    q4_categorical_features(df)
    q5_missingness(df)
    q6_outliers(df)
    print(f"\nPlots saved to {PLOTS}")


if __name__ == "__main__":
    main()
