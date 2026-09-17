"""Data loading, multi-section lookup parsing, cohort filtering, and patient-grouped splitting."""

import csv
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    BINARY_TARGET_COLUMN,
    COLUMNS_TO_DROP,
    DIABETIC_DATA_FILE,
    EXPIRED_DISCHARGE_IDS,
    GROUP_COLUMN,
    IDS_MAPPING_FILE,
    RANDOM_SEED,
    RAW_TARGET_COLUMN,
    TEST_SPLIT_SIZE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_ids_mapping(filepath: Path = IDS_MAPPING_FILE) -> Dict[str, Dict[int, str]]:
    """Parse the multi-section IDS_mapping.csv into separate lookup dictionaries.
    
    The file contains three stacked tables:
      1. admission_type_id
      2. discharge_disposition_id
      3. admission_source_id
    """
    if not filepath.exists():
        raise FileNotFoundError(f"IDS mapping file not found at: {filepath}")

    mappings: Dict[str, Dict[int, str]] = {
        "admission_type": {},
        "discharge_disposition": {},
        "admission_source": {},
    }

    current_section = None

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not any(row):
                current_section = None
                continue

            first_col = row[0].strip()
            if first_col == "admission_type_id":
                current_section = "admission_type"
                continue
            elif first_col == "discharge_disposition_id":
                current_section = "discharge_disposition"
                continue
            elif first_col == "admission_source_id":
                current_section = "admission_source"
                continue

            if current_section and len(row) >= 2:
                id_str = row[0].strip()
                desc = row[1].strip()
                if id_str.isdigit():
                    mappings[current_section][int(id_str)] = desc

    logger.info("Successfully loaded IDS mapping sections: %s", list(mappings.keys()))
    return mappings


def load_raw_data(filepath: Path = DIABETIC_DATA_FILE) -> pd.DataFrame:
    """Load the raw diabetic_data.csv dataset."""
    if not filepath.exists():
        raise FileNotFoundError(f"Raw diabetic data file not found at: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)
    logger.info("Loaded raw diabetic data with shape: %s", df.shape)
    return df


def clean_and_filter_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and apply clinical eligibility cohort filters.
    
    1. Removes expired encounters (discharge_disposition_id in [11, 19, 20, 21])
       because readmission is biologically impossible and creates false negative bias.
    2. Removes invalid gender entries ('Unknown/Invalid').
    3. Creates binary 30-day readmission target (1 if '<30' else 0).
    4. Drops zero-variance and non-feature columns (examide, citoglipton, weight, encounter_id).
    5. Replaces '?' with NaN for standard downstream processing.
    """
    initial_rows = len(df)

    # 1. Mortality filtering
    df = df.copy()
    expired_mask = df["discharge_disposition_id"].isin(EXPIRED_DISCHARGE_IDS)
    expired_count = int(expired_mask.sum())
    df = df[~expired_mask].copy()
    logger.info("Filtered out %d expired encounters (discharge IDs %s)", expired_count, EXPIRED_DISCHARGE_IDS)

    # 2. Filter invalid demographic records
    invalid_gender_mask = df["gender"] == "Unknown/Invalid"
    invalid_gender_count = int(invalid_gender_mask.sum())
    df = df[~invalid_gender_mask].copy()
    logger.info("Filtered out %d invalid gender encounters", invalid_gender_count)

    # 3. Create binary readmission target
    if RAW_TARGET_COLUMN in df.columns:
        df[BINARY_TARGET_COLUMN] = (df[RAW_TARGET_COLUMN] == "<30").astype(int)
        pos_count = int(df[BINARY_TARGET_COLUMN].sum())
        pos_pct = (pos_count / len(df)) * 100
        logger.info(
            "Created binary target '%s': %d positive (<30d) encounters (%.2f%%)",
            BINARY_TARGET_COLUMN,
            pos_count,
            pos_pct,
        )

    # 4. Standardize missingness symbol '?'
    df = df.replace("?", np.nan)

    # 5. Drop zero-variance & non-feature columns if present
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop_cols)
    logger.info("Dropped columns: %s. Remaining columns: %d", drop_cols, len(df.columns))

    logger.info("Cleaned cohort shape: %s (retained %.2f%% of initial records)", df.shape, (len(df) / initial_rows) * 100)
    return df


def create_patient_grouped_split(
    df: pd.DataFrame,
    test_size: float = TEST_SPLIT_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partition dataset into an 80/20 train/test split grouped strictly by patient_nbr.
    
    Guarantees zero patient leakage between the train and test partitions.
    """
    if GROUP_COLUMN not in df.columns:
        raise KeyError(f"Group column '{GROUP_COLUMN}' not found in dataframe.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[GROUP_COLUMN]

    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Assert zero patient overlap
    train_patients = set(train_df[GROUP_COLUMN])
    test_patients = set(test_df[GROUP_COLUMN])
    patient_intersection = train_patients.intersection(test_patients)

    if patient_intersection:
        raise ValueError(
            f"Patient leakage detected! {len(patient_intersection)} patients overlap between train and test."
        )

    logger.info("Patient-grouped split created successfully:")
    logger.info("  Train set: %d encounters across %d unique patients", len(train_df), len(train_patients))
    logger.info("  Test set:  %d encounters across %d unique patients", len(test_df), len(test_patients))
    logger.info(
        "  Train positive rate: %.2f%% | Test positive rate: %.2f%%",
        (train_df[BINARY_TARGET_COLUMN].mean() * 100),
        (test_df[BINARY_TARGET_COLUMN].mean() * 100),
    )

    return train_df, test_df
