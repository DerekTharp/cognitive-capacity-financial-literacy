"""IV/2SLS analysis for cognitive capacity and cognitive decline (Table 3).

Reports matched-sample first stage, reduced form, IV, and OLS estimates.
If an instrument is weak (F <= 10), the script reports the reduced form but
skips the IV estimate instead of stopping the full pipeline.
"""

import json
import logging

import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from statsmodels.tools import add_constant

from src.config import (
    ANALYTIC_PATH, AGE_LOWER_BOUND, CAPACITY_IV_VAR, DECLINE_IV_VAR,
    CAPACITY_ENDOG, DECLINE_ENDOG, MR_COVARIATES, OUTPUT_TABLES,
)

logger = logging.getLogger(__name__)

COVAR_FORMULA = " + ".join(MR_COVARIATES)
WEAK_F_THRESHOLD = 10


def _append_result(results, model, variable, fitted, extra=None):
    ci = fitted.conf_int().loc[variable]
    row = {
        "model": model,
        "variable": variable,
        "coef": fitted.params[variable],
        "se": fitted.bse[variable],
        "pvalue": fitted.pvalues[variable],
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "n": int(fitted.nobs),
        "r2": fitted.rsquared,
    }
    if extra:
        row.update(extra)
    results.append(row)


def _run_single_endogenous_arm(df, endogenous, instrument, label, results):
    needed = ["big3_pct", endogenous, instrument] + MR_COVARIATES
    sub = df.dropna(subset=needed).copy()
    logger.info("%s sample (age %d+): N=%d", label, AGE_LOWER_BOUND, len(sub))

    # First stage
    first_stage = smf.ols(
        f"{endogenous} ~ {instrument} + {COVAR_FORMULA}",
        data=sub,
    ).fit(cov_type="HC3")
    fs_f = float((first_stage.params[instrument] / first_stage.bse[instrument]) ** 2)
    logger.info(
        "%s first stage: coef=%.3f, F=%.1f, p=%.2e",
        label, first_stage.params[instrument], fs_f, first_stage.pvalues[instrument],
    )
    _append_result(
        results,
        f"First stage: {label}",
        instrument,
        first_stage,
        extra={"f_stat": fs_f},
    )

    # Reduced form on the matched IV sample
    reduced_form = smf.ols(
        f"big3_pct ~ {instrument} + {COVAR_FORMULA}",
        data=sub,
    ).fit(cov_type="HC3")
    logger.info(
        "%s reduced form: coef=%.3f (se=%.3f), p=%.4f",
        label, reduced_form.params[instrument], reduced_form.bse[instrument],
        reduced_form.pvalues[instrument],
    )
    _append_result(results, f"Reduced form: {label}", instrument, reduced_form)

    iv_row = None
    if fs_f > WEAK_F_THRESHOLD:
        exog = add_constant(sub[MR_COVARIATES])
        iv_mod = IV2SLS(
            dependent=sub["big3_pct"],
            exog=exog,
            endog=sub[[endogenous]],
            instruments=sub[[instrument]],
        )
        iv_res = iv_mod.fit(cov_type="robust")
        iv_ci = iv_res.conf_int().loc[endogenous]
        iv_row = {
            "model": "IV/2SLS" if label == "capacity" else f"IV/2SLS: {label}",
            "variable": f"{endogenous} (instrumented)",
            "coef": iv_res.params[endogenous],
            "se": iv_res.std_errors[endogenous],
            "pvalue": iv_res.pvalues[endogenous],
            "ci_lower": iv_ci.iloc[0],
            "ci_upper": iv_ci.iloc[1],
            "n": int(iv_res.nobs),
            "r2": iv_res.rsquared,
        }
        results.append(iv_row)
        logger.info(
            "%s IV/2SLS: coef=%.3f (se=%.3f), p=%.4f, [%.3f, %.3f]",
            label,
            iv_row["coef"], iv_row["se"], iv_row["pvalue"],
            iv_row["ci_lower"], iv_row["ci_upper"],
        )
    else:
        logger.warning(
            "%s instrument is weak (F=%.1f <= %d); reporting reduced form only",
            label, fs_f, WEAK_F_THRESHOLD,
        )

    ols = smf.ols(
        f"big3_pct ~ {endogenous} + {COVAR_FORMULA}",
        data=sub,
    ).fit(cov_type="HC3")
    logger.info(
        "%s OLS: coef=%.3f (se=%.3f), p=%.4f",
        label, ols.params[endogenous], ols.bse[endogenous], ols.pvalues[endogenous],
    )
    _append_result(
        results,
        "OLS" if label == "capacity" else f"OLS: {label}",
        endogenous,
        ols,
    )

    return {
        "n": len(sub),
        "first_stage_f": fs_f,
        "iv_row": iv_row,
        "ols_coef": float(ols.params[endogenous]),
        "reduced_form_coef": float(reduced_form.params[instrument]),
    }


def run_iv_analysis(df):
    """Run IV/2SLS and OLS comparisons on the pre-wave trajectory subsample."""
    df_60 = df[df["age_at_wave"] >= AGE_LOWER_BOUND].copy()
    df_xs = (df_60.sort_values("wave_year")
             .drop_duplicates("hhidpn", keep="first"))

    results = []

    # Capacity arm: cognition PGS -> cognitive level -> financial literacy
    capacity = _run_single_endogenous_arm(
        df_xs,
        endogenous=CAPACITY_ENDOG,
        instrument=CAPACITY_IV_VAR,
        label="capacity",
        results=results,
    )

    # Decline arm: Alzheimer's PGS -> cognitive decline -> financial literacy
    decline = _run_single_endogenous_arm(
        df_xs,
        endogenous=DECLINE_ENDOG,
        instrument=DECLINE_IV_VAR,
        label="decline",
        results=results,
    )

    # OLS decomposition on the joint sample
    joint_needed = ["big3_pct", CAPACITY_ENDOG, DECLINE_ENDOG] + MR_COVARIATES
    df_joint = df_xs.dropna(subset=joint_needed).copy()
    ols_both = smf.ols(
        f"big3_pct ~ {CAPACITY_ENDOG} + {DECLINE_ENDOG} + {COVAR_FORMULA}",
        data=df_joint,
    ).fit(cov_type="HC3")
    for var in [CAPACITY_ENDOG, DECLINE_ENDOG]:
        _append_result(results, "OLS: level + decline", var, ols_both)
        logger.info(
            "Joint OLS %s: coef=%.3f (se=%.3f), p=%.4f",
            var, ols_both.params[var], ols_both.bse[var], ols_both.pvalues[var],
        )

    summary = {
        "n_capacity": capacity["n"],
        "capacity_first_stage_F": round(capacity["first_stage_f"], 1),
        "capacity_reduced_form_coef": round(capacity["reduced_form_coef"], 4),
        "capacity_ols_coef": round(capacity["ols_coef"], 4),
        "capacity_iv_coef": (
            round(float(capacity["iv_row"]["coef"]), 4)
            if capacity["iv_row"] is not None else None
        ),
        "n_decline": decline["n"],
        "decline_first_stage_F": round(decline["first_stage_f"], 1),
        "decline_reduced_form_coef": round(decline["reduced_form_coef"], 4),
        "decline_ols_coef": round(decline["ols_coef"], 4),
        "decline_iv_coef": (
            round(float(decline["iv_row"]["coef"]), 4)
            if decline["iv_row"] is not None else None
        ),
        "joint_level_coef": round(float(ols_both.params[CAPACITY_ENDOG]), 4),
        "joint_decline_coef": round(float(ols_both.params[DECLINE_ENDOG]), 4),
        "joint_decline_pvalue": round(float(ols_both.pvalues[DECLINE_ENDOG]), 4),
    }
    summary_path = OUTPUT_TABLES / "iv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", summary_path)

    return pd.DataFrame(results)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_parquet(ANALYTIC_PATH)
    results = run_iv_analysis(df)
    out = OUTPUT_TABLES / "table3_iv.csv"
    results.to_csv(out, index=False)
    logger.info("Saved: %s", out)
    return results


if __name__ == "__main__":
    main()
