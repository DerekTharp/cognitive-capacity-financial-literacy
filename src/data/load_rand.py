"""Load RAND HRS longitudinal data: demographics and cognition by wave."""

import logging

import pandas as pd

from src.config import (
    RAND_PATH, RAND_DEMOGRAPHICS, COGNITION_WAVES, RAND_WAVE_YEARS,
    COG_MEASURES, COGNITION_LONG_PATH,
)

logger = logging.getLogger(__name__)


def load_demographics():
    """Load time-invariant demographics: hhidpn, rabyear, male, educ_years."""
    df = pd.read_stata(str(RAND_PATH), columns=RAND_DEMOGRAPHICS)
    df["hhidpn"] = df["hhidpn"].astype(int)
    df["rabyear"] = pd.to_numeric(df["rabyear"], errors="coerce")
    df["male"] = df["ragender"].astype(str).str.startswith("1.").astype(int)
    df["educ_years"] = pd.to_numeric(df["raedyrs"], errors="coerce")
    assert df["hhidpn"].is_unique, "Duplicate hhidpn in RAND demographics"
    logger.info("Demographics: N=%d", len(df))
    return df[["hhidpn", "rabyear", "male", "educ_years"]]


def load_cognition_long():
    """Extract cognitive measures across waves 3-13 in long format.

    Loads both COG27 (Langa-Weir 27-point composite) and TR20 (word recall)
    for each wave. Age computed as wave_year - rabyear.
    """
    cog_cols = []
    for measure_key, spec in COG_MEASURES.items():
        for w in COGNITION_WAVES:
            cog_cols.append(spec["var_template"].format(w=w))

    all_cols = ["hhidpn", "rabyear"] + cog_cols
    df_wide = pd.read_stata(str(RAND_PATH), columns=all_cols)
    df_wide["hhidpn"] = df_wide["hhidpn"].astype(int)
    df_wide["rabyear"] = pd.to_numeric(df_wide["rabyear"], errors="coerce")

    frames = []
    for w in COGNITION_WAVES:
        sub = df_wide[["hhidpn", "rabyear"]].copy()
        sub["wave"] = w
        sub["year"] = RAND_WAVE_YEARS[w]
        sub["age"] = sub["year"] - sub["rabyear"]

        for measure_key, spec in COG_MEASURES.items():
            col = spec["var_template"].format(w=w)
            sub[measure_key] = pd.to_numeric(df_wide[col], errors="coerce")

        # Require at least one cognitive measure and age to keep the row
        has_any_cog = sub[list(COG_MEASURES.keys())].notna().any(axis=1)
        sub = sub[has_any_cog & sub["age"].notna()].copy()
        frames.append(sub[["hhidpn", "wave", "year", "age"] + list(COG_MEASURES.keys())])

    df_long = pd.concat(frames, ignore_index=True)
    logger.info(
        "Cognition long: %d person-waves, %d unique respondents, measures: %s",
        len(df_long), df_long["hhidpn"].nunique(), list(COG_MEASURES.keys()),
    )
    return df_long


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    demo = load_demographics()
    cog = load_cognition_long()
    cog.to_parquet(COGNITION_LONG_PATH, index=False)
    logger.info("Saved: %s", COGNITION_LONG_PATH)
    return demo, cog


if __name__ == "__main__":
    main()
