#!/usr/bin/env python3
"""Master script: regenerates all outputs from raw data.

Usage:
    python3 run_all.py              # Full pipeline (data build + analysis)
    python3 run_all.py --skip-data  # Use cached .parquet files, re-run analysis only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import OUTPUT_TABLES, OUTPUT_FIGURES, ANALYTIC_ALL_PATH, ANALYTIC_PATH

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run full analysis pipeline")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data building; use cached parquet files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ---- Step 1: Build analytic sample ----
    if not args.skip_data or not ANALYTIC_PATH.exists() or not ANALYTIC_ALL_PATH.exists():
        logger.info("=" * 60)
        logger.info("STEP 1: Building analytic sample")
        logger.info("=" * 60)
        from src.data.merge import build_analytic_sample
        df_60, df_all = build_analytic_sample(
            include_sensitivity_pgs=True,
            force_rebuild=True,
            return_all=True,
        )
        df_all.to_parquet(ANALYTIC_ALL_PATH, index=False)
        df_60.to_parquet(ANALYTIC_PATH, index=False)
    else:
        logger.info("Using cached analytic sample: %s", ANALYTIC_PATH)

    # ---- Step 2: Descriptives ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Descriptive statistics")
    logger.info("=" * 60)
    from src.analysis.descriptives import main as run_descriptives
    run_descriptives()

    # ---- Step 3: Reduced-form MR ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Reduced-form MR")
    logger.info("=" * 60)
    from src.analysis.reduced_form import main as run_reduced_form
    run_reduced_form()

    # ---- Step 4: IV/2SLS ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: IV/2SLS analysis")
    logger.info("=" * 60)
    from src.analysis.iv_analysis import main as run_iv
    run_iv()

    # ---- Step 5: Trajectory analysis ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Trajectory analysis")
    logger.info("=" * 60)
    from src.analysis.trajectory_analysis import main as run_trajectories
    run_trajectories()

    # ---- Step 6: Wave-by-wave replication ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Wave-by-wave replication")
    logger.info("=" * 60)
    from src.analysis.wave_replication import main as run_wave
    run_wave()

    # ---- Step 7: Within-person change ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Within-person change analysis")
    logger.info("=" * 60)
    from src.analysis.within_person import main as run_within
    run_within()

    # ---- Step 8: Sensitivity analyses ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: Sensitivity analyses")
    logger.info("=" * 60)
    from src.analysis.sensitivity import main as run_sensitivity
    run_sensitivity()

    # ---- Step 9: Supplementary (AFR/HIS) ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 9: Supplementary ancestry analyses")
    logger.info("=" * 60)
    from src.analysis.supplementary import main as run_supplementary
    run_supplementary()

    # ---- Step 10: Figures ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 10: Figures")
    logger.info("=" * 60)
    from src.figures import main as run_figures
    run_figures()

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("DONE. Total time: %.0f seconds", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
