"""Trajectory analysis: cognitive level vs decline as predictors (Table 4).

Supporting identification strategy: person-specific cognitive trajectories
(random-intercept, random-slope) decompose the age-financial literacy
gradient into level versus decline components. Slope variables are coded
so higher values mean faster decline.
"""

import logging

import pandas as pd
import statsmodels.formula.api as smf

from src.config import (
    ANALYTIC_PATH, AGE_LOWER_BOUND, PC_VARS, OUTPUT_TABLES,
    CAPACITY_ENDOG, DECLINE_ENDOG,
)

logger = logging.getLogger(__name__)


def run_trajectory_analysis(df):
    """Test whether cognitive level or decline predicts financial literacy."""
    df_60 = df[df["age_at_wave"] >= AGE_LOWER_BOUND].copy()
    df_xs = (df_60.sort_values("wave_year")
             .drop_duplicates("hhidpn", keep="first"))

    level = CAPACITY_ENDOG
    slope = DECLINE_ENDOG
    traj_vars = ["big3_pct", level, slope, "rabyear", "male", "age_at_wave"]
    df_t = df_xs.dropna(subset=traj_vars)
    logger.info("Trajectory sample (age %d+): N=%d", AGE_LOWER_BOUND, len(df_t))

    pc_str = " + ".join(PC_VARS)
    covars = f"rabyear + male + {pc_str}"

    specs = [
        ("Level only",
         f"big3_pct ~ {level} + {covars}",
         [level]),
        ("Decline only",
         f"big3_pct ~ {slope} + {covars}",
         [slope]),
        ("Level + decline",
         f"big3_pct ~ {level} + {slope} + {covars}",
         [level, slope]),
        ("Level + decline + age",
         f"big3_pct ~ {level} + {slope} + age_at_wave + {covars}",
         [level, slope, "age_at_wave"]),
    ]

    results = []
    for label, formula, key_vars in specs:
        m = smf.ols(formula, data=df_t).fit(cov_type="HC3")
        logger.info("\n%s (N=%d, R2=%.4f):", label, int(m.nobs), m.rsquared)
        for var in key_vars:
            ci = m.conf_int().loc[var]
            r = {
                "model": label, "variable": var,
                "coef": m.params[var], "se": m.bse[var], "pvalue": m.pvalues[var],
                "ci_lower": ci[0], "ci_upper": ci[1],
                "n": int(m.nobs), "r2": m.rsquared,
            }
            results.append(r)
            logger.info("  %-20s %8.3f (%6.3f)  p=%.4f", var, r["coef"], r["se"], r["pvalue"])

    return pd.DataFrame(results)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_parquet(ANALYTIC_PATH)
    results = run_trajectory_analysis(df)
    out = OUTPUT_TABLES / "table4_trajectories.csv"
    results.to_csv(out, index=False)
    logger.info("Saved: %s", out)
    return results


if __name__ == "__main__":
    main()
