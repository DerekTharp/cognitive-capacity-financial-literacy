"""Supplementary analysis: reduced-form MR in African and Hispanic ancestry samples.

PGS discovery GWAS are predominantly European, so predictive power is
expected to be lower in non-European populations due to ancestry mismatch.
These results are exploratory and reported for transparency.
"""

import logging

import pandas as pd
import statsmodels.formula.api as smf

from src.config import AGE_LOWER_BOUND, MR_COVARIATES, OUTPUT_TABLES
from src.data.merge import build_ancestry_sample

logger = logging.getLogger(__name__)

COVAR_FORMULA = " + ".join(MR_COVARIATES)


def _run_ancestry(ancestry):
    """Run reduced-form MR for one non-European ancestry."""
    df = build_ancestry_sample(ancestry)

    df_60 = df[df["age_at_wave"] >= AGE_LOWER_BOUND].copy()
    mr_vars = ["big3_pct", "pgs_cognition", "pgs_alz_wa"] + MR_COVARIATES
    df_60 = df_60.dropna(subset=mr_vars)

    df_xs = (df_60.sort_values("wave_year")
             .drop_duplicates("hhidpn", keep="first"))

    logger.info("%s cross-sectional (age %d+): N=%d", ancestry, AGE_LOWER_BOUND, len(df_xs))

    if len(df_xs) < 50:
        logger.warning("%s: N=%d too small for regression, skipping", ancestry, len(df_xs))
        return pd.DataFrame()

    results = []
    specs = [
        ("PGS cognition",
         f"big3_pct ~ pgs_cognition + {COVAR_FORMULA}",
         ["pgs_cognition"]),
        ("PGS Alzheimer's (WA)",
         f"big3_pct ~ pgs_alz_wa + {COVAR_FORMULA}",
         ["pgs_alz_wa"]),
        ("PGS cognition + Alzheimer's",
         f"big3_pct ~ pgs_cognition + pgs_alz_wa + {COVAR_FORMULA}",
         ["pgs_cognition", "pgs_alz_wa"]),
    ]

    for label, formula, key_vars in specs:
        m = smf.ols(formula, data=df_xs).fit(cov_type="HC3")
        logger.info("  %s %s (N=%d, R2=%.4f):", ancestry, label, int(m.nobs), m.rsquared)
        for var in key_vars:
            ci = m.conf_int().loc[var]
            r = {
                "ancestry": ancestry,
                "model": label,
                "variable": var,
                "coef": m.params[var],
                "se": m.bse[var],
                "pvalue": m.pvalues[var],
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "n": int(m.nobs),
                "r2": m.rsquared,
            }
            results.append(r)
            logger.info("    %-20s %8.3f (%6.3f)  p=%.4f",
                        var, r["coef"], r["se"], r["pvalue"])

    return pd.DataFrame(results)


def run_supplementary():
    """Run reduced-form MR for African and Hispanic ancestry samples."""
    frames = []
    for ancestry in ["AFR", "HIS"]:
        logger.info("\n--- %s ancestry ---", ancestry)
        result = _run_ancestry(ancestry)
        if not result.empty:
            frames.append(result)

    if not frames:
        logger.warning("No supplementary ancestry results produced")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = run_supplementary()
    out = OUTPUT_TABLES / "tableS2_ancestry.csv"
    results.to_csv(out, index=False)
    logger.info("Saved: %s", out)
    return results


if __name__ == "__main__":
    main()
