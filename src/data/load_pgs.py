"""Load HRS Polygenic Scores (Release 5) by ancestry with validation."""

import logging

import pandas as pd

from src.config import (
    PGS_FILES, ANCESTRY_PREFIX, EXPECTED_PGS_N,
    PGS_LEVEL, PGS_DECLINE, PGS_SENSITIVITY, AUX_PGS, PC_VARS,
)

logger = logging.getLogger(__name__)


def _build_hhidpn(df):
    hhid = df["HHID"].astype(str).str.strip()
    pn = df["PN"].astype(str).str.strip().str.zfill(3)
    return (hhid + pn).astype(int)


def _verify_standardisation(df, cols, ancestry):
    for col in cols:
        mean, std = df[col].mean(), df[col].std()
        assert abs(mean) < 0.05, f"{ancestry} {col}: mean={mean:.4f}, expected ~0"
        assert abs(std - 1.0) < 0.10, f"{ancestry} {col}: SD={std:.4f}, expected ~1"


def load_pgs(ancestry="EUR", include_sensitivity=False):
    """Load PGS data for a given ancestry.

    Returns DataFrame with hhidpn, pgs_* columns, and PC1-PC10.
    """
    path = PGS_FILES[ancestry]
    prefix = ANCESTRY_PREFIX[ancestry]

    col_map = {}
    for out_name, suffix in {**PGS_LEVEL, **PGS_DECLINE}.items():
        col_map[f"{prefix}_{suffix}"] = out_name
    if include_sensitivity:
        for out_name, suffix in PGS_SENSITIVITY.items():
            col_map[f"{prefix}_{suffix}"] = out_name
        for out_name, suffix in AUX_PGS.items():
            col_map[f"{prefix}_{suffix}"] = out_name

    stata_cols = ["HHID", "PN"] + list(col_map.keys()) + PC_VARS
    df = pd.read_stata(str(path), columns=stata_cols)
    df["hhidpn"] = _build_hhidpn(df)

    assert df["hhidpn"].is_unique, f"Duplicate hhidpn in {ancestry} PGS"

    expected = EXPECTED_PGS_N[ancestry]
    assert abs(len(df) - expected) / expected < 0.02, (
        f"{ancestry} PGS: N={len(df)}, expected ~{expected}"
    )

    df = df.rename(columns=col_map)
    pgs_cols = list(col_map.values())
    _verify_standardisation(df, pgs_cols, ancestry)

    for pc in PC_VARS:
        assert df[pc].notna().all(), f"{ancestry}: {pc} has missing values"

    logger.info("%s PGS: N=%d", ancestry, len(df))
    return df[["hhidpn"] + pgs_cols + PC_VARS].copy()


def load_pgs_european(include_sensitivity=False):
    return load_pgs("EUR", include_sensitivity=include_sensitivity)


def assert_ancestry_nonoverlap():
    ids = {}
    for anc in ["EUR", "AFR", "HIS"]:
        df = pd.read_stata(str(PGS_FILES[anc]), columns=["HHID", "PN"])
        df["hhidpn"] = _build_hhidpn(df)
        ids[anc] = set(df["hhidpn"])
    for a, b in [("EUR", "AFR"), ("EUR", "HIS"), ("AFR", "HIS")]:
        overlap = ids[a] & ids[b]
        assert len(overlap) == 0, f"{a}/{b} overlap: {len(overlap)} respondents"
    logger.info("Ancestry non-overlap verified")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pgs = load_pgs_european(include_sensitivity=True)
    assert_ancestry_nonoverlap()
    return pgs


if __name__ == "__main__":
    main()
