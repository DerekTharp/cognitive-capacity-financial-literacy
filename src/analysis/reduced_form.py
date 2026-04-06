"""Reduced-form Mendelian randomisation: PGS → financial literacy (Table 2)."""

import logging

import pandas as pd
import statsmodels.formula.api as smf

from src.config import (
    ANALYTIC_ALL_PATH, ANALYTIC_PATH, AGE_LOWER_BOUND,
    MR_COVARIATES, PC_VARS, OUTPUT_TABLES,
)

logger = logging.getLogger(__name__)

COVAR_FORMULA = " + ".join(MR_COVARIATES)


def _fit(formula, data, label, key_vars, cluster=None):
    kw = ({"cov_type": "cluster", "cov_kwds": {"groups": data[cluster]}}
          if cluster else {"cov_type": "HC3"})
    m = smf.ols(formula, data=data).fit(**kw)

    rows = []
    logger.info("%s (N=%d, R2=%.4f)", label, int(m.nobs), m.rsquared)
    for var in key_vars:
        if var not in m.params.index:
            continue
        ci = m.conf_int().loc[var]
        r = {
            "model": label, "variable": var,
            "coef": m.params[var], "se": m.bse[var], "pvalue": m.pvalues[var],
            "ci_lower": ci[0], "ci_upper": ci[1],
            "n": int(m.nobs), "r2": m.rsquared,
        }
        rows.append(r)
        logger.info("  %-30s %8.3f (%6.3f)  p=%.4f  [%.3f, %.3f]",
                    var, r["coef"], r["se"], r["pvalue"], r["ci_lower"], r["ci_upper"])
    return rows, m


def run_reduced_form(df, df_all=None):
    """Run all reduced-form specifications on EUR genotyped, age 60+."""
    df_60 = df[df["age_at_wave"] >= AGE_LOWER_BOUND].copy()
    df_xs = (df_60.sort_values("wave_year")
             .drop_duplicates("hhidpn", keep="first"))
    mr_vars = ["big3_pct", "pgs_cognition", "pgs_alz_wa"] + MR_COVARIATES
    df_xs = df_xs.dropna(subset=mr_vars)

    logger.info("Cross-sectional sample (age %d+): N=%d", AGE_LOWER_BOUND, len(df_xs))

    all_rows = []

    # Model 1: PGS cognition only
    r, _ = _fit(f"big3_pct ~ pgs_cognition + {COVAR_FORMULA}",
                df_xs, "PGS cognition", ["pgs_cognition"])
    all_rows.extend(r)

    # Model 2: PGS Alzheimer's only
    r, _ = _fit(f"big3_pct ~ pgs_alz_wa + {COVAR_FORMULA}",
                df_xs, "PGS Alzheimer's (WA)", ["pgs_alz_wa"])
    all_rows.extend(r)

    # Model 3: PGS cognition + Alzheimer's
    r, _ = _fit(f"big3_pct ~ pgs_cognition + pgs_alz_wa + {COVAR_FORMULA}",
                df_xs, "PGS cognition + Alzheimer's",
                ["pgs_cognition", "pgs_alz_wa"])
    all_rows.extend(r)

    # Model 4: Age gradient
    r, _ = _fit(f"big3_pct ~ age_at_wave + {COVAR_FORMULA}",
                df_xs, "Age gradient", ["age_at_wave"])
    all_rows.extend(r)

    # All ages (no age restriction)
    if df_all is not None:
        df_all_xs = (df_all.sort_values("wave_year")
                     .drop_duplicates("hhidpn", keep="first"))
        df_all_xs = df_all_xs.dropna(subset=mr_vars)
        r, _ = _fit(f"big3_pct ~ pgs_cognition + pgs_alz_wa + {COVAR_FORMULA}",
                    df_all_xs, "All ages: PGS cognition + Alzheimer's",
                    ["pgs_cognition", "pgs_alz_wa"])
        all_rows.extend(r)

    # Pooled panel with clustered SEs and wave fixed effects
    df_panel = df_60.dropna(subset=mr_vars)
    pc_str = " + ".join(PC_VARS)
    panel_cov = f"rabyear + male + C(wave_year) + {pc_str}"
    r, _ = _fit(f"big3_pct ~ pgs_cognition + pgs_alz_wa + {panel_cov}",
                df_panel, "Pooled panel (clustered)",
                ["pgs_cognition", "pgs_alz_wa"], cluster="hhidpn")
    all_rows.extend(r)

    return pd.DataFrame(all_rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_parquet(ANALYTIC_PATH)
    df_all = pd.read_parquet(ANALYTIC_ALL_PATH) if ANALYTIC_ALL_PATH.exists() else None
    results = run_reduced_form(df, df_all=df_all)
    out = OUTPUT_TABLES / "table2_reduced_form.csv"
    results.to_csv(out, index=False)
    logger.info("Saved: %s", out)
    return results


if __name__ == "__main__":
    main()
