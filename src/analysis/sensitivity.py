"""Sensitivity analyses: alternative PGS instruments (Supplementary Table S1).

Tests robustness of main results to alternative GWAS sources for both
cognition PGS (CHARGE 2015 vs 2018) and Alzheimer's PGS (PGC 2021 vs
IGAP 2013/2019, EADB 2022; with vs without APOE region).
"""

import logging

import pandas as pd
import statsmodels.formula.api as smf

from src.config import ANALYTIC_PATH, AGE_LOWER_BOUND, MR_COVARIATES, PC_VARS, OUTPUT_TABLES

logger = logging.getLogger(__name__)

COVAR_FORMULA = " + ".join(MR_COVARIATES)


def run_sensitivity(df):
    """Run reduced-form MR with alternative PGS instruments."""
    df_60 = df[df["age_at_wave"] >= AGE_LOWER_BOUND].copy()
    df_xs = (df_60.sort_values("wave_year")
             .drop_duplicates("hhidpn", keep="first"))

    results = []

    # --- Alternative Alzheimer's PGS ---
    alz_variants = [
        ("pgs_alz_wa", "PGC 2021 with APOE [primary]"),
        ("pgs_alz_na", "PGC 2021 no APOE"),
        ("pgs_alz_eadb22_wa", "EADB 2022 with APOE"),
        ("pgs_alz_eadb22_na", "EADB 2022 no APOE"),
        ("pgs_alz_igap19_wa", "IGAP 2019 with APOE"),
        ("pgs_alz_igap19_na", "IGAP 2019 no APOE"),
        ("pgs_alz_igap13_wa", "IGAP 2013 with APOE"),
        ("pgs_alz_igap13_na", "IGAP 2013 no APOE"),
    ]

    logger.info("--- Alternative Alzheimer's PGS ---")
    for pgs_var, label in alz_variants:
        if pgs_var not in df_xs.columns:
            logger.info("  %s: not in data, skipping", label)
            continue

        needed = ["big3_pct", "pgs_cognition", pgs_var] + MR_COVARIATES
        sub = df_xs.dropna(subset=needed)
        if len(sub) < 50:
            continue

        m = smf.ols(
            f"big3_pct ~ pgs_cognition + {pgs_var} + {COVAR_FORMULA}",
            data=sub,
        ).fit(cov_type="HC3")

        for var in ["pgs_cognition", pgs_var]:
            ci = m.conf_int().loc[var]
            results.append({
                "analysis": f"Alt Alz: {label}",
                "variable": var,
                "coef": m.params[var], "se": m.bse[var], "pvalue": m.pvalues[var],
                "ci_lower": ci[0], "ci_upper": ci[1],
                "n": int(m.nobs), "r2": m.rsquared,
            })

        logger.info("  %s (N=%d): cog=%.3f (p=%.4f), alz=%.3f (p=%.4f)",
                    label, int(m.nobs),
                    m.params["pgs_cognition"], m.pvalues["pgs_cognition"],
                    m.params[pgs_var], m.pvalues[pgs_var])

    # --- Alternative cognition PGS (CHARGE 2015) ---
    logger.info("\n--- Alternative Cognition PGS ---")
    if "pgs_cognition_2015" in df_xs.columns:
        needed = ["big3_pct", "pgs_cognition_2015", "pgs_alz_wa"] + MR_COVARIATES
        sub = df_xs.dropna(subset=needed)
        m = smf.ols(
            f"big3_pct ~ pgs_cognition_2015 + pgs_alz_wa + {COVAR_FORMULA}",
            data=sub,
        ).fit(cov_type="HC3")

        for var in ["pgs_cognition_2015", "pgs_alz_wa"]:
            ci = m.conf_int().loc[var]
            results.append({
                "analysis": "Alt cognition: CHARGE 2015",
                "variable": var,
                "coef": m.params[var], "se": m.bse[var], "pvalue": m.pvalues[var],
                "ci_lower": ci[0], "ci_upper": ci[1],
                "n": int(m.nobs), "r2": m.rsquared,
            })
        logger.info("  CHARGE 2015 (N=%d): cog=%.3f (p=%.4f)",
                    int(m.nobs), m.params["pgs_cognition_2015"],
                    m.pvalues["pgs_cognition_2015"])

    # --- Education PGS adjustment (sensitivity only) ---
    logger.info("\n--- Education PGS Adjustment ---")
    needed = ["big3_pct", "pgs_cognition", "pgs_education", "pgs_alz_wa"] + MR_COVARIATES
    sub = df_xs.dropna(subset=needed)
    if len(sub) >= 50:
        m = smf.ols(
            f"big3_pct ~ pgs_cognition + pgs_education + pgs_alz_wa + {COVAR_FORMULA}",
            data=sub,
        ).fit(cov_type="HC3")
        for var in ["pgs_cognition", "pgs_education", "pgs_alz_wa"]:
            ci = m.conf_int().loc[var]
            results.append({
                "analysis": "Education PGS adjustment",
                "variable": var,
                "coef": m.params[var], "se": m.bse[var], "pvalue": m.pvalues[var],
                "ci_lower": ci[0], "ci_upper": ci[1],
                "n": int(m.nobs), "r2": m.rsquared,
            })
        logger.info(
            "  Education PGS adjustment (N=%d): cog=%.3f (p=%.4f), "
            "edu=%.3f (p=%.4f), alz=%.3f (p=%.4f)",
            int(m.nobs),
            m.params["pgs_cognition"], m.pvalues["pgs_cognition"],
            m.params["pgs_education"], m.pvalues["pgs_education"],
            m.params["pgs_alz_wa"], m.pvalues["pgs_alz_wa"],
        )

    # --- Height PGS falsification ---
    logger.info("\n--- Height PGS Falsification ---")
    if "pgs_height" in df_xs.columns:
        needed = ["big3_pct", "pgs_height"] + MR_COVARIATES
        sub = df_xs.dropna(subset=needed)
        m = smf.ols(
            f"big3_pct ~ pgs_height + {COVAR_FORMULA}",
            data=sub,
        ).fit(cov_type="HC3")
        ci = m.conf_int().loc["pgs_height"]
        results.append({
            "analysis": "Height PGS falsification",
            "variable": "pgs_height",
            "coef": m.params["pgs_height"], "se": m.bse["pgs_height"],
            "pvalue": m.pvalues["pgs_height"],
            "ci_lower": ci[0], "ci_upper": ci[1],
            "n": int(m.nobs), "r2": m.rsquared,
        })
        logger.info("  Height PGS (N=%d): b=%.3f (p=%.4f)",
                    int(m.nobs), m.params["pgs_height"], m.pvalues["pgs_height"])

    # --- Directionality check (partial R² comparison) ---
    logger.info("\n--- Directionality Check ---")
    from src.config import CAPACITY_ENDOG
    traj_vars = [CAPACITY_ENDOG, "big3_pct", "pgs_cognition"] + MR_COVARIATES
    sub = df_xs.dropna(subset=traj_vars)
    if len(sub) >= 50:
        m_exp = smf.ols(f"{CAPACITY_ENDOG} ~ pgs_cognition + {COVAR_FORMULA}", data=sub).fit()
        m_out = smf.ols(f"big3_pct ~ pgs_cognition + {COVAR_FORMULA}", data=sub).fit()

        r2_exposure = m_exp.rsquared - smf.ols(
            f"{CAPACITY_ENDOG} ~ {COVAR_FORMULA}", data=sub).fit().rsquared
        r2_outcome = m_out.rsquared - smf.ols(
            f"big3_pct ~ {COVAR_FORMULA}", data=sub).fit().rsquared
        correct_direction = r2_exposure > r2_outcome

        results.append({
            "analysis": "Directionality check",
            "variable": "pgs_cognition",
            "coef": float("nan"), "se": float("nan"), "pvalue": float("nan"),
            "ci_lower": float("nan"), "ci_upper": float("nan"),
            "n": int(m_exp.nobs), "r2": float("nan"),
            "partial_r2_exposure": r2_exposure,
            "partial_r2_outcome": r2_outcome,
            "correct_direction": correct_direction,
        })
        logger.info("  Partial R2: PGS→exposure=%.4f, PGS→outcome=%.4f, correct=%s",
                    r2_exposure, r2_outcome, correct_direction)

    # --- Covariate balance ---
    logger.info("\n--- Covariate Balance ---")
    pc_str = " + ".join(PC_VARS)
    for covar, label in [("rabyear", "Birth year"), ("male", "Male")]:
        sub = df_xs.dropna(subset=[covar, "pgs_cognition"] + PC_VARS)
        m = smf.ols(f"{covar} ~ pgs_cognition + {pc_str}", data=sub).fit(cov_type="HC3")
        results.append({
            "analysis": f"Balance: {label}",
            "variable": "pgs_cognition",
            "coef": m.params["pgs_cognition"], "se": m.bse["pgs_cognition"],
            "pvalue": m.pvalues["pgs_cognition"],
            "ci_lower": float("nan"), "ci_upper": float("nan"),
            "n": int(m.nobs), "r2": m.rsquared,
        })
        logger.info("  PGS cognition → %s: b=%.4f (p=%.4f)", label,
                    m.params["pgs_cognition"], m.pvalues["pgs_cognition"])

    return pd.DataFrame(results)


def run_tr20_sensitivity(df):
    """TR20 (word recall) sensitivity: parallel IV and trajectory results."""
    from linearmodels.iv import IV2SLS
    from statsmodels.tools import add_constant
    from src.config import CAPACITY_IV_VAR, DECLINE_IV_VAR

    df_60 = df[df["age_at_wave"] >= AGE_LOWER_BOUND].copy()
    df_xs = (df_60.sort_values("wave_year")
             .drop_duplicates("hhidpn", keep="first"))

    results = []

    # TR20 capacity first stage + IV
    needed = ["big3_pct", "tr20_level", "tr20_slope", CAPACITY_IV_VAR] + MR_COVARIATES
    sub = df_xs.dropna(subset=needed).copy()
    if len(sub) >= 50:
        fs = smf.ols(
            f"tr20_level ~ {CAPACITY_IV_VAR} + {COVAR_FORMULA}", data=sub,
        ).fit(cov_type="HC3")
        fs_f = float((fs.params[CAPACITY_IV_VAR] / fs.bse[CAPACITY_IV_VAR]) ** 2)

        results.append({
            "analysis": "TR20 sensitivity", "variable": "first_stage_F",
            "coef": fs_f, "se": float("nan"), "pvalue": fs.pvalues[CAPACITY_IV_VAR],
            "ci_lower": float("nan"), "ci_upper": float("nan"),
            "n": int(fs.nobs), "r2": fs.rsquared,
        })

        if fs_f > 10:
            exog = add_constant(sub[MR_COVARIATES])
            iv_mod = IV2SLS(
                dependent=sub["big3_pct"], exog=exog,
                endog=sub[["tr20_level"]], instruments=sub[[CAPACITY_IV_VAR]],
            )
            iv_res = iv_mod.fit(cov_type="robust")
            iv_ci = iv_res.conf_int().loc["tr20_level"]
            results.append({
                "analysis": "TR20 sensitivity", "variable": "IV_tr20_level",
                "coef": iv_res.params["tr20_level"],
                "se": iv_res.std_errors["tr20_level"],
                "pvalue": iv_res.pvalues["tr20_level"],
                "ci_lower": iv_ci.iloc[0], "ci_upper": iv_ci.iloc[1],
                "n": int(iv_res.nobs), "r2": iv_res.rsquared,
            })

        ols = smf.ols(
            f"big3_pct ~ tr20_level + {COVAR_FORMULA}", data=sub,
        ).fit(cov_type="HC3")
        ci = ols.conf_int().loc["tr20_level"]
        results.append({
            "analysis": "TR20 sensitivity", "variable": "OLS_tr20_level",
            "coef": ols.params["tr20_level"], "se": ols.bse["tr20_level"],
            "pvalue": ols.pvalues["tr20_level"],
            "ci_lower": ci[0], "ci_upper": ci[1],
            "n": int(ols.nobs), "r2": ols.rsquared,
        })

        # TR20 trajectory decomposition
        ols_both = smf.ols(
            f"big3_pct ~ tr20_level + tr20_slope + {COVAR_FORMULA}", data=sub,
        ).fit(cov_type="HC3")
        for var in ["tr20_level", "tr20_slope"]:
            ci = ols_both.conf_int().loc[var]
            results.append({
                "analysis": "TR20 sensitivity", "variable": f"OLS_joint_{var}",
                "coef": ols_both.params[var], "se": ols_both.bse[var],
                "pvalue": ols_both.pvalues[var],
                "ci_lower": ci[0], "ci_upper": ci[1],
                "n": int(ols_both.nobs), "r2": ols_both.rsquared,
            })

        logger.info("TR20 sensitivity: F=%.1f, IV=%.3f, OLS=%.3f, N=%d",
                    fs_f,
                    results[1]["coef"] if len(results) > 1 else float("nan"),
                    ols.params["tr20_level"], int(ols.nobs))

    return pd.DataFrame(results)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_parquet(ANALYTIC_PATH)
    results = run_sensitivity(df)
    out = OUTPUT_TABLES / "tableS1_sensitivity.csv"
    results.to_csv(out, index=False)
    logger.info("Saved: %s", out)

    tr20 = run_tr20_sensitivity(df)
    tr20_out = OUTPUT_TABLES / "tableS3_tr20_sensitivity.csv"
    tr20.to_csv(tr20_out, index=False)
    logger.info("Saved: %s", tr20_out)
    return results


if __name__ == "__main__":
    main()
