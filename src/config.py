"""Central configuration: all paths, variable definitions, and parameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HRS_DIR = PROJECT_ROOT / "HRS"
FAT_DIR = HRS_DIR / "HRS Fat Files"
DATA_DIR = PROJECT_ROOT / "data"
RESTRICTED_DIR = DATA_DIR / "raw" / "restricted"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_TABLES = OUTPUT_DIR / "tables"
OUTPUT_FIGURES = OUTPUT_DIR / "figures"

RAND_PATH = HRS_DIR / "randhrs1992_2022v1_STATA" / "randhrs1992_2022v1.dta"

FAT_FILES = {
    2004: FAT_DIR / "h04f1c_STATA" / "h04f1c.dta",
    2010: FAT_DIR / "hd10f6b_STATA" / "hd10f6b.dta",
    2016: FAT_DIR / "h16f2c_STATA" / "h16f2c.dta",
    2020: FAT_DIR / "h20f1b_STATA" / "h20f1b.dta",
}

PGS_FILES = {
    "EUR": RESTRICTED_DIR / "PGENSCOREE_R.dta",
    "AFR": RESTRICTED_DIR / "PGENSCOREA_R.dta",
    "HIS": RESTRICTED_DIR / "PGENSCOREH_R.dta",
}

# Intermediate data
ANALYTIC_ALL_PATH = DATA_DIR / "analytic_sample_all_ages.parquet"
ANALYTIC_PATH = DATA_DIR / "analytic_sample.parquet"
TRAJECTORIES_PATH = DATA_DIR / "cognitive_trajectories.parquet"
COGNITION_LONG_PATH = DATA_DIR / "cognition_long.parquet"

# ---------------------------------------------------------------------------
# PGS configuration (HRS Polygenic Scores Release 5)
# ---------------------------------------------------------------------------
ANCESTRY_PREFIX = {"EUR": "E5", "AFR": "A5", "HIS": "H5"}
EXPECTED_PGS_N = {"EUR": 12090, "AFR": 3100, "HIS": 2381}

# Level instruments — suffix after ancestry prefix
PGS_LEVEL = {
    "pgs_cognition": "GENCOG2_CHARGE18",
    "pgs_education": "EDU3_SSGAC18",
}

# Decline instruments — primary (PGC 2021 Alzheimer's GWAS)
PGS_DECLINE = {
    "pgs_alz_wa": "GWALZWA_PGC21",    # with APOE region
    "pgs_alz_na": "GWALZNA_PGC21",    # without APOE region
}

# Auxiliary PGS for falsification
AUX_PGS = {
    "pgs_height": "HEIGHT2_GIANT18",
}

# Sensitivity PGS variants
PGS_SENSITIVITY = {
    "pgs_cognition_2015": "GENCOG_CHARGE15",
    "pgs_alz_eadb22_wa": "GWPROXYALZWA_EADB22",
    "pgs_alz_eadb22_na": "GWPROXYALZNA_EADB22",
    "pgs_alz_igap19_wa": "GWAD2WA_IGAP19",
    "pgs_alz_igap19_na": "GWAD2NA_IGAP19",
    "pgs_alz_igap13_wa": "GWADWA_IGAP13",
    "pgs_alz_igap13_na": "GWADNA_IGAP13",
}

# Ancestry principal components (all 10 required in every PGS regression)
PC_VARS = [
    "PC1_5A", "PC1_5B", "PC1_5C", "PC1_5D", "PC1_5E",
    "PC6_10A", "PC6_10B", "PC6_10C", "PC6_10D", "PC6_10E",
]

# ---------------------------------------------------------------------------
# Financial literacy scoring by wave
# ---------------------------------------------------------------------------
# Each item: (variable_name_or_list, correct_value)
# DK=8 and RF=9 scored as incorrect (0), following Lusardi-Mitchell convention.
# 2016 uses 1=True/2=False (KNOWN DEVIATION from 1=True/5=False in other waves).
# 2020 scoring verified against the official HRS 2020 Module 8 questionnaire.
INCLUDE_2020_WAVE = True

BIG3_SCORING = {
    2004: {
        "compound_interest": ("jv364", 1),      # "More than $102"
        "inflation":         ("jv365", 3),      # "Less"
        "diversification":   ("jv363", 5),      # "False" (mutual fund safer)
    },
    2010: {
        "compound_interest": ("mv351", 1),
        "inflation":         ("mv352", 3),
        "diversification":   ("mv353", 5),
    },
    2016: {
        "compound_interest": (["pv052", "pv102"], 1),   # split forms A/B
        "inflation":         (["pv053", "pv103"], 3),
        "diversification":   (["pv054", "pv104"], 2),   # 2="False" in 2016 coding
    },
    2020: {
        "compound_interest": ("rv565", 3),      # "More than $102"
        "inflation":         ("rv566", 3),      # "Less than today"
        "diversification":   ("rv567", 5),      # released fat file uses 5="False"
    },
}

FAT_COLUMNS = {
    2004: ["hhid", "pn", "jv363", "jv364", "jv365"],
    2010: ["hhid", "pn", "mv351", "mv352", "mv353"],
    2016: ["hhid", "pn", "pv052", "pv053", "pv054", "pv102", "pv103", "pv104"],
    2020: ["hhid", "pn", "rv565", "rv566", "rv567"],
}

FINLIT_WAVES = [2004, 2010, 2016] + ([2020] if INCLUDE_2020_WAVE else [])
FINLIT_ITEMS = ["compound_interest", "inflation", "diversification"]

# ---------------------------------------------------------------------------
# RAND longitudinal variables
# ---------------------------------------------------------------------------
RAND_DEMOGRAPHICS = ["hhidpn", "rabyear", "ragender", "raedyrs"]

# Cognitive measures available waves 3-13 (1996-2016).
# Waves 14-15 split into phone/web mode; wave 16 not yet released.
COGNITION_WAVES = list(range(3, 14))
RAND_WAVE_YEARS = {
    3: 1996, 4: 1998, 5: 2000, 6: 2002, 7: 2004,
    8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014,
    13: 2016,
}

# Primary: COG27 (Langa-Weir 27-point: word recall + serial 7s + backwards counting).
# Better construct alignment with CHARGE PGS (general cognitive function).
# Sensitivity: TR20 (word recall 0-20, episodic memory only).
COG_MEASURES = {
    "cog27": {"var_template": "r{w}cog27", "label": "General cognition (0-27)", "scale_max": 27},
    "tr20":  {"var_template": "r{w}tr20",  "label": "Word recall (0-20)",       "scale_max": 20},
}
PRIMARY_COG_MEASURE = "cog27"

# ---------------------------------------------------------------------------
# Trajectory estimation
# ---------------------------------------------------------------------------
AGE_CENTER = 70
MIN_WAVES_FOR_SLOPE = 3

# ---------------------------------------------------------------------------
# Sample restrictions
# ---------------------------------------------------------------------------
AGE_LOWER_BOUND = 60

# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------
# MR covariates: birth year, gender, PCs 1-10.
# Do NOT add education, cognition, or financial variables — on the causal pathway.
MR_COVARIATES = ["rabyear", "male"] + PC_VARS

CAPACITY_IV_VAR = "pgs_cognition"
DECLINE_IV_VAR = "pgs_alz_wa"

# Trajectory variable names derived from PRIMARY_COG_MEASURE
CAPACITY_ENDOG = f"{PRIMARY_COG_MEASURE}_level"   # e.g. "cog27_level"
DECLINE_ENDOG = f"{PRIMARY_COG_MEASURE}_slope"     # e.g. "cog27_slope"

# Sensitivity: TR20 trajectory names
SENSITIVITY_ENDOG_LEVEL = "tr20_level"
SENSITIVITY_ENDOG_SLOPE = "tr20_slope"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 20260405
