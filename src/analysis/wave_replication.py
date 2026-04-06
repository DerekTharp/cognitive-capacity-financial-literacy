"""Wave-by-wave replication: PGS -> financial literacy in each wave (Table 5).

Tests whether the cognition PGS result replicates across the available,
verified HRS experimental waves.
"""

import logging

import pandas as pd
import statsmodels.formula.api as smf

from src.config import ANALYTIC_PATH, AGE_LOWER_BOUND, MR_COVARIATES, OUTPUT_TABLES

logger = logging.getLogger(__name__)

COVAR_FORMULA = " + ".join(MR_COVARIATES)


def run_wave_replication(df):
    """Run PGS → finlit regressions separately by wave."""
    df_60 = df[df["age_at_wave"] >= AGE_LOWER_BOUND].copy()
    mr_vars = ["big3_pct", "pgs_cognition", "pgs_alz_wa"] + MR_COVARIATES
    df_60 = df_60.dropna(subset=mr_vars)

    results = []
    for year in sorted(df_60["wave_year"].unique()):
        sub = df_60[df_60["wave_year"] == year]
        if len(sub) < 50:
            logger.warning("Wave %d: N=%d, skipping", year, len(sub))
            continue

        m = smf.ols(
            f"big3_pct ~ pgs_cognition + pgs_alz_wa + {COVAR_FORMULA}",
            data=sub,
        ).fit(cov_type="HC3")

        logger.info("Wave %d (N=%d, R2=%.4f):", year, int(m.nobs), m.rsquared)
        for var in ["pgs_cognition", "pgs_alz_wa"]:
            ci = m.conf_int().loc[var]
            r = {
                "wave": year, "variable": var,
                "coef": m.params[var], "se": m.bse[var], "pvalue": m.pvalues[var],
                "ci_lower": ci[0], "ci_upper": ci[1],
                "n": int(m.nobs), "r2": m.rsquared,
            }
            results.append(r)
            logger.info("  %-20s %8.3f (%6.3f)  p=%.4f",
                        var, r["coef"], r["se"], r["pvalue"])

    return pd.DataFrame(results)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_parquet(ANALYTIC_PATH)
    results = run_wave_replication(df)
    out = OUTPUT_TABLES / "table5_wave_replication.csv"
    results.to_csv(out, index=False)
    logger.info("Saved: %s", out)
    return results


if __name__ == "__main__":
    main()
