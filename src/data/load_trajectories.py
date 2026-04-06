"""Estimate pre-wave cognitive trajectories via mixed-effects models.

For each financial-literacy wave and each cognitive measure, we estimate
respondent-specific trajectories using only observations that precede that wave.

Model: cog_it = (b0 + u0i) + (b1 + u1i) * age_centered_it + e_it

Age is centred at 70. Exported measures per cognitive variable:
  {measure}_level = predicted cognition at age 70
  {measure}_slope = negated annual slope (higher = faster decline)
"""

import logging

import pandas as pd
import statsmodels.formula.api as smf

from src.config import AGE_CENTER, MIN_WAVES_FOR_SLOPE, FINLIT_WAVES, COG_MEASURES

logger = logging.getLogger(__name__)


def _fit_wave_measure_model(df_long, wave_year, measure_key):
    """Estimate pre-wave trajectories for one wave and one cognitive measure."""
    df = df_long[df_long["year"] < wave_year].dropna(subset=[measure_key, "age"]).copy()
    df["age_c"] = df["age"] - AGE_CENTER

    waves_per_person = df.groupby("hhidpn").size()
    eligible = waves_per_person[waves_per_person >= MIN_WAVES_FOR_SLOPE].index
    df = df[df["hhidpn"].isin(eligible)].copy()

    level_col = f"{measure_key}_level"
    slope_col = f"{measure_key}_slope"

    if df.empty or len(eligible) == 0:
        logger.warning("Wave %d %s: no eligible respondents", wave_year, measure_key)
        return pd.DataFrame(
            columns=["hhidpn", "wave_year", level_col, slope_col, f"{measure_key}_n_waves"]
        )

    logger.info(
        "Trajectory %s for %d: %d persons, %d+ pre-wave waves (%d obs)",
        measure_key, wave_year, len(eligible), MIN_WAVES_FOR_SLOPE, len(df),
    )

    model = smf.mixedlm(
        f"{measure_key} ~ age_c",
        data=df,
        groups=df["hhidpn"],
        re_formula="~age_c",
    )
    result = model.fit(reml=True, method="lbfgs")

    logger.info(
        "  Fixed effects: level=%.3f, raw slope=%.4f/yr",
        result.fe_params["Intercept"], result.fe_params["age_c"],
    )

    rows = []
    for pid, effects in result.random_effects.items():
        raw_slope = result.fe_params["age_c"] + effects["age_c"]
        rows.append({
            "hhidpn": int(pid),
            "wave_year": wave_year,
            level_col: result.fe_params["Intercept"] + effects["Group"],
            slope_col: -raw_slope,
        })

    trajectories = pd.DataFrame(rows)
    n_waves = waves_per_person.loc[eligible].rename(f"{measure_key}_n_waves").reset_index()
    trajectories = trajectories.merge(n_waves, on="hhidpn", validate="one_to_one")
    return trajectories


def estimate_trajectories(df_long):
    """Estimate pre-wave trajectories for each wave and each cognitive measure."""
    all_frames = []
    for wave_year in FINLIT_WAVES:
        wave_parts = []
        for measure_key in COG_MEASURES:
            part = _fit_wave_measure_model(df_long, wave_year, measure_key)
            wave_parts.append(part)

        # Merge all measures for this wave on (hhidpn, wave_year)
        merged = wave_parts[0]
        for part in wave_parts[1:]:
            merged = merged.merge(part, on=["hhidpn", "wave_year"], how="outer")
        all_frames.append(merged)

    trajectories = pd.concat(all_frames, ignore_index=True)
    assert trajectories.duplicated(["hhidpn", "wave_year"]).sum() == 0
    logger.info("Trajectories: %d person-wave rows, measures: %s",
                len(trajectories), list(COG_MEASURES.keys()))
    return trajectories


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from src.config import COGNITION_LONG_PATH, TRAJECTORIES_PATH

    if COGNITION_LONG_PATH.exists():
        df_long = pd.read_parquet(COGNITION_LONG_PATH)
    else:
        from src.data.load_rand import load_cognition_long
        df_long = load_cognition_long()
        df_long.to_parquet(COGNITION_LONG_PATH, index=False)

    trajectories = estimate_trajectories(df_long)
    trajectories.to_parquet(TRAJECTORIES_PATH, index=False)
    logger.info("Saved: %s", TRAJECTORIES_PATH)
    return trajectories


if __name__ == "__main__":
    main()
