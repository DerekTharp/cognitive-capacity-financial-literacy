"""Descriptive statistics for the analytic sample (Table 1)."""

import logging

import numpy as np
import pandas as pd

from src.config import ANALYTIC_PATH, AGE_LOWER_BOUND, OUTPUT_TABLES

logger = logging.getLogger(__name__)


def run_descriptives(df):
    """Generate Table 1: sample characteristics for EUR genotyped, age 60+."""
    df_60 = df[df["age_at_wave"] >= AGE_LOWER_BOUND].copy()
    df_xs = (df_60.sort_values("wave_year")
             .drop_duplicates("hhidpn", keep="first"))

    stats = []

    def _add(label, values, fmt="continuous"):
        if fmt == "continuous":
            stats.append({
                "variable": label, "mean": values.mean(), "sd": values.std(),
                "min": values.min(), "max": values.max(), "n": int(values.notna().sum()),
            })
        elif fmt == "pct":
            stats.append({
                "variable": label, "mean": values.mean() * 100, "sd": np.nan,
                "min": 0, "max": 100, "n": int(values.notna().sum()),
            })

    _add("Age at wave", df_xs["age_at_wave"])
    _add("Male", df_xs["male"], "pct")
    _add("Education (years)", df_xs["educ_years"])
    _add("Birth year", df_xs["rabyear"])
    _add("Big 3 score (0-3)", df_xs["big3_score"])
    _add("Big 3 percent correct", df_xs["big3_pct"])
    _add("Compound interest correct", df_xs["compound_interest"], "pct")
    _add("Inflation correct", df_xs["inflation"], "pct")
    _add("Diversification correct", df_xs["diversification"], "pct")
    _add("PGS cognition (SD units)", df_xs["pgs_cognition"])
    _add("PGS education (SD units)", df_xs["pgs_education"])
    _add("PGS Alzheimer's WA (SD units)", df_xs["pgs_alz_wa"])

    from src.config import CAPACITY_ENDOG, DECLINE_ENDOG, PRIMARY_COG_MEASURE, COG_MEASURES
    measure_label = COG_MEASURES[PRIMARY_COG_MEASURE]["label"]
    df_traj = df_xs.dropna(subset=[CAPACITY_ENDOG])
    _add(f"Cognitive level at 70 ({measure_label})", df_traj[CAPACITY_ENDOG])
    _add(f"Cognitive decline rate ({measure_label}/yr)", df_traj[DECLINE_ENDOG])
    stats[-2]["n_traj"] = len(df_traj)
    stats[-1]["n_traj"] = len(df_traj)

    result = pd.DataFrame(stats)

    logger.info("--- Table 1: Descriptives (age %d+, N=%d) ---",
                AGE_LOWER_BOUND, len(df_xs))
    for _, row in result.iterrows():
        if pd.isna(row["sd"]):
            logger.info("  %-40s  %.1f%% (N=%d)", row["variable"], row["mean"], row["n"])
        else:
            logger.info("  %-40s  %.2f (SD=%.2f, N=%d)",
                        row["variable"], row["mean"], row["sd"], row["n"])

    logger.info("\nWave distribution:\n%s",
                df_xs["wave_year"].value_counts().sort_index().to_string())

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_parquet(ANALYTIC_PATH)
    result = run_descriptives(df)
    out = OUTPUT_TABLES / "table1_descriptives.csv"
    result.to_csv(out, index=False)
    logger.info("Saved: %s", out)
    return result


if __name__ == "__main__":
    main()
