"""Within-person analysis: does PGS predict annualised financial literacy change?

Tests whether genetic predisposition to cognition or Alzheimer's predicts
within-person change in financial literacy across waves. The dependent
variable is annualised change so respondents with different follow-up lengths
are comparable.
"""

import logging

import pandas as pd
import statsmodels.formula.api as smf

from src.config import ANALYTIC_PATH, AGE_LOWER_BOUND, MR_COVARIATES, OUTPUT_TABLES

logger = logging.getLogger(__name__)

COVAR_FORMULA = " + ".join(MR_COVARIATES)


def run_within_person(df):
    """Test whether PGS predicts within-person financial literacy change."""
    df_60 = df[df["age_at_wave"] >= AGE_LOWER_BOUND].copy()
    mr_vars = ["big3_pct", "pgs_cognition", "pgs_alz_wa"] + MR_COVARIATES
    df_60 = df_60.dropna(subset=mr_vars)

    multi = df_60.groupby("hhidpn").size()
    multi_ids = multi[multi >= 2].index
    df_multi = df_60[df_60["hhidpn"].isin(multi_ids)].copy()

    logger.info("Respondents with 2+ waves: %d (%d person-waves)",
                len(multi_ids), len(df_multi))

    if len(multi_ids) < 50:
        logger.warning("Too few multi-wave respondents for within-person analysis")
        return pd.DataFrame()

    df_sorted = df_multi.sort_values("wave_year")
    df_first = df_sorted.drop_duplicates("hhidpn", keep="first").set_index("hhidpn")
    df_last = df_sorted.drop_duplicates("hhidpn", keep="last").set_index("hhidpn")

    df_change = pd.DataFrame({
        "delta_finlit": df_last["big3_pct"] - df_first["big3_pct"],
        "years_between": df_last["wave_year"] - df_first["wave_year"],
        "baseline_big3": df_first["big3_pct"],
    })
    df_change.index.name = "hhidpn"
    df_change = df_change.reset_index()
    df_change = df_change[df_change["years_between"] > 0].copy()
    df_change["delta_finlit_annual"] = (
        df_change["delta_finlit"] / df_change["years_between"]
    )

    # Merge time-invariant PGS and demographics from first observation
    pgs_demo_cols = (["pgs_cognition", "pgs_alz_wa", "pgs_education",
                      "rabyear", "male"]
                     + [c for c in df_first.columns if c.startswith("PC")])
    pgs_demo = df_first[pgs_demo_cols]
    df_change = df_change.merge(pgs_demo, left_on="hhidpn", right_index=True)

    logger.info("Mean years between obs: %.1f", df_change["years_between"].mean())
    logger.info("Mean finlit change: %.1f pp total (SD=%.1f)",
                df_change["delta_finlit"].mean(), df_change["delta_finlit"].std())
    logger.info("Mean annualised finlit change: %.2f pp/year (SD=%.2f)",
                df_change["delta_finlit_annual"].mean(),
                df_change["delta_finlit_annual"].std())

    # Log ceiling-effect diagnostics
    n_perfect = (df_change["baseline_big3"] >= 100).sum()
    logger.info("Perfect score (100%%) at baseline: %d (%.1f%%)",
                n_perfect, 100 * n_perfect / len(df_change))

    specs = [
        ("PGS cognition -> annualised delta finlit",
         f"delta_finlit_annual ~ pgs_cognition + {COVAR_FORMULA}",
         ["pgs_cognition"]),
        ("PGS Alzheimer's -> annualised delta finlit",
         f"delta_finlit_annual ~ pgs_alz_wa + {COVAR_FORMULA}",
         ["pgs_alz_wa"]),
        ("PGS cog + Alz -> annualised delta finlit",
         f"delta_finlit_annual ~ pgs_cognition + pgs_alz_wa + {COVAR_FORMULA}",
         ["pgs_cognition", "pgs_alz_wa"]),
        ("PGS cognition -> delta (baseline-controlled)",
         f"delta_finlit_annual ~ pgs_cognition + baseline_big3 + {COVAR_FORMULA}",
         ["pgs_cognition", "baseline_big3"]),
        ("PGS cog + Alz -> delta (baseline-controlled)",
         f"delta_finlit_annual ~ pgs_cognition + pgs_alz_wa + baseline_big3 + {COVAR_FORMULA}",
         ["pgs_cognition", "pgs_alz_wa", "baseline_big3"]),
    ]

    results = []
    for label, formula, key_vars in specs:
        needed = ["delta_finlit_annual"] + key_vars + list(MR_COVARIATES)
        sub = df_change.dropna(subset=needed)
        m = smf.ols(formula, data=sub).fit(cov_type="HC3")
        logger.info("\n%s (N=%d, R2=%.4f):", label, int(m.nobs), m.rsquared)

        for var in key_vars:
            ci = m.conf_int().loc[var]
            r = {
                "model": label, "variable": var,
                "coef": m.params[var], "se": m.bse[var], "pvalue": m.pvalues[var],
                "ci_lower": ci[0], "ci_upper": ci[1],
                "n": int(m.nobs), "r2": m.rsquared,
                "mean_years_between": df_change["years_between"].mean(),
            }
            results.append(r)
            logger.info("  %-20s %8.3f (%6.3f)  p=%.4f",
                        var, r["coef"], r["se"], r["pvalue"])

    return pd.DataFrame(results)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_parquet(ANALYTIC_PATH)
    results = run_within_person(df)
    out = OUTPUT_TABLES / "table6_within_person.csv"
    results.to_csv(out, index=False)
    logger.info("Saved: %s", out)
    return results


if __name__ == "__main__":
    main()
