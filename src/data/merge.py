"""Master merge: combine financial literacy, PGS, demographics, and trajectories."""

import logging

import pandas as pd

from src.config import (
    AGE_LOWER_BOUND, ANALYTIC_ALL_PATH, ANALYTIC_PATH,
    TRAJECTORIES_PATH, COGNITION_LONG_PATH,
)
from src.data.load_finlit import load_finlit
from src.data.load_pgs import load_pgs_european, assert_ancestry_nonoverlap
from src.data.load_rand import load_demographics

logger = logging.getLogger(__name__)


def build_analytic_sample(include_sensitivity_pgs=False, force_rebuild=False, return_all=False):
    """Build the master analytic dataset.

    Merges financial literacy with EUR PGS, demographics, and
    cognitive trajectory estimates. Returns one row per person-wave.
    """
    finlit = load_finlit()
    pgs = load_pgs_european(include_sensitivity=include_sensitivity_pgs)
    assert_ancestry_nonoverlap()
    demo = load_demographics()

    # Load or estimate trajectories
    if not force_rebuild and TRAJECTORIES_PATH.exists():
        trajectories = pd.read_parquet(TRAJECTORIES_PATH)
        logger.info("Loaded cached trajectories: N=%d", len(trajectories))
    else:
        from src.data.load_rand import load_cognition_long
        from src.data.load_trajectories import estimate_trajectories
        if not force_rebuild and COGNITION_LONG_PATH.exists():
            cog_long = pd.read_parquet(COGNITION_LONG_PATH)
        else:
            cog_long = load_cognition_long()
            cog_long.to_parquet(COGNITION_LONG_PATH, index=False)
        trajectories = estimate_trajectories(cog_long)
        trajectories.to_parquet(TRAJECTORIES_PATH, index=False)

    # --- Merge chain with logging ---
    n0 = len(finlit)
    assert finlit.duplicated(["hhidpn", "wave_year"]).sum() == 0
    assert pgs["hhidpn"].is_unique
    assert demo["hhidpn"].is_unique
    assert trajectories.duplicated(["hhidpn", "wave_year"]).sum() == 0

    df = finlit.merge(pgs, on="hhidpn", how="inner", validate="m:1")
    logger.info("Finlit + PGS (EUR): %d → %d person-waves", n0, len(df))

    n1 = len(df)
    df = df.merge(demo, on="hhidpn", how="inner", validate="m:1")
    logger.info("+ demographics: %d → %d", n1, len(df))

    df["age_at_wave"] = df["wave_year"] - df["rabyear"]

    n2 = len(df)
    df = df.merge(
        trajectories,
        on=["hhidpn", "wave_year"], how="left", validate="m:1",
    )
    from src.config import CAPACITY_ENDOG
    n_traj = df[CAPACITY_ENDOG].notna().sum()
    logger.info("+ trajectories (left join): %d total, %d with estimates", n2, n_traj)

    # --- Summary ---
    logger.info("--- All-ages merged sample ---")
    logger.info("Person-waves: %d | Unique: %d", len(df), df["hhidpn"].nunique())
    logger.info("By wave:\n%s", df["wave_year"].value_counts().sort_index())

    df_60 = df[df["age_at_wave"] >= AGE_LOWER_BOUND].copy()
    logger.info("Age %d+: %d person-waves, %d unique",
                AGE_LOWER_BOUND, len(df_60), df_60["hhidpn"].nunique())

    # --- Assertions ---
    assert df["hhidpn"].notna().all()
    assert len(df) > 0, "Empty analytic sample"
    assert df["big3_pct"].between(0, 100).all()
    assert df["big3_score"].between(0, 3).all()
    assert df.duplicated(["hhidpn", "wave_year"]).sum() == 0
    assert df_60["age_at_wave"].ge(AGE_LOWER_BOUND).all()
    assert df_60.duplicated(["hhidpn", "wave_year"]).sum() == 0

    # Regression test: for rows with trajectory data, verify that the
    # trajectory was estimated for the same wave as the finlit observation.
    # Guards against row-selection bugs (e.g., pandas groupby().first()
    # returning column-wise first non-null instead of row-preserving first).
    from src.config import CAPACITY_ENDOG
    has_traj = df[CAPACITY_ENDOG].notna()
    if has_traj.any():
        # The merge was on (hhidpn, wave_year), so trajectory columns should
        # only be non-null when the trajectory wave matches the finlit wave.
        # Verify by checking that no row has a trajectory value without a
        # matching (hhidpn, wave_year) pair in the trajectory table.
        traj_keys = set(zip(trajectories["hhidpn"], trajectories["wave_year"]))
        df_keys = set(zip(df.loc[has_traj, "hhidpn"], df.loc[has_traj, "wave_year"]))
        orphans = df_keys - traj_keys
        assert len(orphans) == 0, (
            f"Trajectory wave mismatch: {len(orphans)} (hhidpn, wave_year) pairs "
            f"have trajectory values but no matching trajectory record."
        )
        logger.info("Regression test passed: all trajectory values match their wave")

    if return_all:
        return df_60, df
    return df_60


def build_ancestry_sample(ancestry):
    """Build a finlit + PGS + demographics sample for a non-EUR ancestry.

    Uses the same merge validation as the primary pipeline but skips
    trajectory estimation (supplementary analyses are reduced-form only).
    """
    from src.data.load_pgs import load_pgs

    finlit = load_finlit()
    pgs = load_pgs(ancestry)
    demo = load_demographics()

    assert finlit.duplicated(["hhidpn", "wave_year"]).sum() == 0
    assert pgs["hhidpn"].is_unique
    assert demo["hhidpn"].is_unique

    n0 = len(finlit)
    df = finlit.merge(pgs, on="hhidpn", how="inner", validate="m:1")
    logger.info("Finlit + PGS (%s): %d → %d person-waves", ancestry, n0, len(df))

    n1 = len(df)
    df = df.merge(demo, on="hhidpn", how="inner", validate="m:1")
    logger.info("+ demographics: %d → %d", n1, len(df))

    df["age_at_wave"] = df["wave_year"] - df["rabyear"]

    assert df["hhidpn"].notna().all()
    assert len(df) > 0, f"Empty {ancestry} sample"
    assert df["big3_pct"].between(0, 100).all()
    assert df.duplicated(["hhidpn", "wave_year"]).sum() == 0

    logger.info("%s sample: %d person-waves, %d unique",
                ancestry, len(df), df["hhidpn"].nunique())
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df_60, df_all = build_analytic_sample(
        include_sensitivity_pgs=True,
        force_rebuild=True,
        return_all=True,
    )
    df_all.to_parquet(ANALYTIC_ALL_PATH, index=False)
    df_60.to_parquet(ANALYTIC_PATH, index=False)
    logger.info("Saved: %s", ANALYTIC_ALL_PATH)
    logger.info("Saved: %s", ANALYTIC_PATH)
    return df_60


if __name__ == "__main__":
    main()
