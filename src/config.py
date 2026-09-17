"""Central configuration for Hospital Readmission Intelligence platform."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories (support both data/raw and raw root directory)
RAW_DATA_CANDIDATE_1 = PROJECT_ROOT / "data" / "raw"
RAW_DATA_CANDIDATE_2 = PROJECT_ROOT / "raw"

if (RAW_DATA_CANDIDATE_1 / "diabetic_data.csv").exists():
    RAW_DATA_DIR = RAW_DATA_CANDIDATE_1
else:
    RAW_DATA_DIR = RAW_DATA_CANDIDATE_2

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# File names
DIABETIC_DATA_FILE = RAW_DATA_DIR / "diabetic_data.csv"
IDS_MAPPING_FILE = RAW_DATA_DIR / "IDS_mapping.csv"

CLEANED_DATASET_FILE = PROCESSED_DATA_DIR / "cleaned_cohort.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train_cohort.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test_cohort.csv"

# Modeling & Splitting Hyperparameters
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.20
CV_FOLDS = 5
GROUP_COLUMN = "patient_nbr"
ENCOUNTER_ID_COLUMN = "encounter_id"
RAW_TARGET_COLUMN = "readmitted"
BINARY_TARGET_COLUMN = "readmitted_30d"

# Clinical Cohort Ineligibility Exclusions
# IDs 11, 19, 20, 21 represent expired (deceased) patients who are ineligible for readmission
EXPIRED_DISCHARGE_IDS = [11, 19, 20, 21]

# Zero and near-zero variance columns to drop
COLUMNS_TO_DROP = [
    "encounter_id",       # Identifier / temporal proxy
    "weight",             # 96.86% missing
    "examide",            # 100% constant 'No'
    "citoglipton",        # 100% constant 'No'
]

# Operational Capacity Default Presets
DEFAULT_CAPACITY_PRESETS = {
    "Top 5%": 0.05,
    "Top 10%": 0.10,
    "Top 20%": 0.20
}

# ICD-9 Diagnosis Groupings (Strack et al., 2014 Standard)
# Reference: Strack et al., BioMed Research International (2014), Table 2
ICD9_CATEGORIES = {
    "Circulatory": [(390, 459), (785, 785)],
    "Respiratory": [(460, 519), (786, 786)],
    "Digestive": [(520, 579), (787, 787)],
    "Diabetes": [(250, 250)], # Special float handling for 250.xx
    "Injury": [(800, 999)],
    "Musculoskeletal": [(710, 739)],
    "Genitourinary": [(580, 629), (788, 788)],
    "Neoplasms": [(140, 239)],
}

# Operational Review Workflows
WORKFLOW_CARE_MANAGEMENT = "Care-Management Review"
WORKFLOW_DISCHARGE_PLANNING = "Discharge-Planning Review"
WORKFLOW_FOLLOWUP_COORDINATION = "Follow-Up Coordination Review"
WORKFLOW_ADDITIONAL_RECORD = "Additional Record Review"
