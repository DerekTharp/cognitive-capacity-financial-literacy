"""Load and score Big 3 financial literacy items across HRS waves (2004-2020)."""

import logging

import numpy as np
import pandas as pd

from src.config import (
    FAT_FILES, FAT_COLUMNS, BIG3_SCORING, FINLIT_ITEMS,
    FINLIT_WAVES, INCLUDE_2020_WAVE,
)

logger = logging.getLogger(__name__)


def _build_hhidpn(df):
    hhid = df["hhid"].astype(str).str.strip()
    pn = df["pn"].astype(str).str.strip().str.zfill(3)
    return (hhid + pn).astype(int)


def _score_item(series, correct_val):
    """Score: 1=correct, 0=incorrect/DK/RF, NaN=truly missing."""
    s = pd.to_numeric(series, errors="coerce")
    scored = pd.Series(np.nan, index=s.index)
    responded = s.notna() & ~s.isin([-8])
    scored[responded] = 0.0
    scored[s == correct_val] = 1.0
    return scored


def _load_wave(year):
    path = FAT_FILES[year]
    cols = FAT_COLUMNS[year]
    scoring = BIG3_SCORING[year]

    df = pd.read_stata(str(path), columns=cols)
    df["hhidpn"] = _build_hhidpn(df)

    for item, spec in scoring.items():
        if isinstance(spec[0], list):
            vars_list, correct_val = spec
            scored = _score_item(df[vars_list[0]], correct_val)
            for v in vars_list[1:]:
                scored = scored.fillna(_score_item(df[v], correct_val))
            df[item] = scored
        else:
            var_name, correct_val = spec
            df[item] = _score_item(df[var_name], correct_val)

    has_all = df[FINLIT_ITEMS].notna().all(axis=1)
    df = df[has_all].copy()
    df["big3_score"] = df[FINLIT_ITEMS].sum(axis=1)
    df["big3_pct"] = df["big3_score"] / 3 * 100
    df["wave_year"] = year

    logger.info(
        "Wave %d: N=%d | compound=%.1f%% inflation=%.1f%% "
        "diversification=%.1f%% mean=%.1f%%",
        year, len(df),
        df["compound_interest"].mean() * 100,
        df["inflation"].mean() * 100,
        df["diversification"].mean() * 100,
        df["big3_pct"].mean(),
    )

    return df[["hhidpn", "wave_year", "big3_score", "big3_pct"] + FINLIT_ITEMS]


def load_finlit():
    """Load and score Big 3 across all waves. Returns long-format DataFrame."""
    if not INCLUDE_2020_WAVE and 2020 not in FINLIT_WAVES:
        logger.info("Skipping 2020 Big 3 wave")

    frames = [_load_wave(year) for year in FINLIT_WAVES]
    df = pd.concat(frames, ignore_index=True)
    assert df.duplicated(["hhidpn", "wave_year"]).sum() == 0, (
        "Duplicate respondent-wave rows in financial literacy data"
    )
    logger.info(
        "Pooled: %d person-waves, %d unique respondents",
        len(df), df["hhidpn"].nunique(),
    )
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return load_finlit()


if __name__ == "__main__":
    main()
