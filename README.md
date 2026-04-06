# Cognitive capacity influences financial literacy in older adults: Mendelian randomisation evidence

Replication package for the NHB manuscript by Derek Tharp.

## Environment

- Python 3.12+
- Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Setup

This project uses both public and restricted HRS inputs.

**Public data** (available at https://hrs.isr.umich.edu/):
- RAND HRS Longitudinal File (1992--2022): place at `HRS/randhrs1992_2022v1_STATA/randhrs1992_2022v1.dta`
- HRS Fat Files for 2004, 2010, 2016, 2020: place under `HRS/HRS Fat Files/`

**Restricted data** (requires HRS restricted-data use agreement):
- `data/raw/restricted/PGENSCOREE_R.dta` (European ancestry PGS)
- `data/raw/restricted/PGENSCOREA_R.dta` (African ancestry PGS)
- `data/raw/restricted/PGENSCOREH_R.dta` (Hispanic ancestry PGS)

These files are from HRS PGS Release 5. Applications: https://hrs.isr.umich.edu/data-products/restricted-data

The `HRS/` directory can be a symlink to a shared HRS data folder.

## Run

Full raw-to-output rebuild (builds analytic sample, estimates trajectories, runs all analyses):

```bash
python3 run_all.py
```

Analysis only, using cached `.parquet` files (requires prior full run):

```bash
python3 run_all.py --skip-data
```

**Note:** `--skip-data` requires that `data/analytic_sample.parquet`, `data/analytic_sample_all_ages.parquet`, `data/cognition_long.parquet`, and `data/cognitive_trajectories.parquet` already exist from a prior full run. These are not shipped with the package because they contain restricted-access genetic data.

## Key Intermediate Files

- `data/analytic_sample.parquet` — authoritative age-60+ analytic sample (EUR genotyped)
- `data/analytic_sample_all_ages.parquet` — all-ages merged sample
- `data/cognition_long.parquet` — RAND cognition panel in long format (COG27 + TR20, waves 3--13)
- `data/cognitive_trajectories.parquet` — wave-specific pre-outcome trajectory estimates (COG27 and TR20)

## Output Files

### Tables
- `output/tables/table1_descriptives.csv` — sample characteristics
- `output/tables/table2_reduced_form.csv` — reduced-form MR (PGS → financial literacy)
- `output/tables/table3_iv.csv` — IV/2SLS capacity and decline arms
- `output/tables/table4_trajectories.csv` — cognitive level vs decline as predictors
- `output/tables/table5_wave_replication.csv` — wave-by-wave PGS coefficients
- `output/tables/table6_within_person.csv` — within-person financial literacy change
- `output/tables/tableS1_sensitivity.csv` — alternative PGS, falsification, Steiger, balance
- `output/tables/tableS2_ancestry.csv` — African/Hispanic ancestry supplementary results
- `output/tables/tableS3_tr20_sensitivity.csv` — word-recall-only IV sensitivity
- `output/tables/iv_summary.json` — key IV statistics

### Figures
- `output/figures/figure1_wave_replication.png` — wave-by-wave coefficient plot
- `output/figures/figure2_iv_vs_ols.png` — IV vs OLS forest plot
- `output/figures/figure3_reduced_form_forest.png` — reduced-form forest plot
- `output/figures/figure4_dag.png` — two-arm MR design schematic

## Analysis Contract

- Primary analyses use the European-ancestry PGS sample only (N = 2,207 age 60+).
- Every PGS regression includes ancestry principal components 1--10.
- The saved analytic sample is restricted to respondents aged 60+.
- Trajectories are estimated separately for each financial-literacy wave using only cognition measured before that wave.
- The primary cognitive measure is the Langa-Weir 27-point composite (COG27: word recall + serial 7s + backwards counting). Word recall alone (TR20) is a sensitivity check.
- `cog27_slope` and `tr20_slope` are coded so larger values mean faster decline.
- The within-person analysis includes both raw and baseline-controlled specifications to account for ceiling effects on the bounded Big 3 measure.

## Manuscript-to-Output Crosswalk

| Manuscript claim | Source file | Row/key |
|---|---|---|
| Cognition PGS b = 5.59 | table2_reduced_form.csv | "PGS cognition + Alzheimer's", pgs_cognition |
| Alzheimer's PGS b = -1.45 | table2_reduced_form.csv | "PGS cognition + Alzheimer's", pgs_alz_wa |
| First-stage F = 256 | iv_summary.json | capacity_first_stage_F |
| IV = 6.21 | iv_summary.json | capacity_iv_coef |
| OLS = 3.19 | iv_summary.json | capacity_ols_coef |
| Decline F < 1 | iv_summary.json | decline_first_stage_F (0.5) |
| Trajectory level b = 3.19 | table4_trajectories.csv | "Level + decline", cog27_level |
| Trajectory slope p = 0.32 | table4_trajectories.csv | "Level + decline", cog27_slope |
| Within-person cog p = 0.70 | table6_within_person.csv | row 1 |
| Within-person alz p = 0.86 | table6_within_person.csv | row 2 |
| Within-person cog baseline-adj p = 0.065 | table6_within_person.csv | row 5 |
| TR20 sensitivity F = 166 | tableS3_tr20_sensitivity.csv | first_stage_F |
| Height falsification p = 0.50 | tableS1_sensitivity.csv | "Height PGS falsification" |
