"""Automated unit tests for data ingestion, cohort filtering, and patient-grouped splitting."""

import numpy as np
import pandas as pd
import pytest

from src.config import EXPIRED_DISCHARGE_IDS, GROUP_COLUMN, BINARY_TARGET_COLUMN
from src.data_loader import clean_and_filter_cohort, create_patient_grouped_split, load_ids_mapping


def test_ids_mapping_structure():
    """Verify IDS_mapping.csv parses correctly into 3 distinct dictionary sections."""
    mappings = load_ids_mapping()
    assert "admission_type" in mappings
    assert "discharge_disposition" in mappings
    assert "admission_source" in mappings

    assert len(mappings["admission_type"]) > 0
    assert len(mappings["discharge_disposition"]) > 0
    assert len(mappings["admission_source"]) > 0

    # Assert known IDs are mapped
    assert mappings["admission_type"].get(1) == "Emergency"
    assert mappings["discharge_disposition"].get(11) == "Expired"
    assert "Emergency Room" in mappings["admission_source"].get(7, "")


def test_mortality_exclusion_and_gender_clean():
    """Verify that expired patients and invalid demographics are cleanly removed."""
    sample_data = pd.DataFrame({
        "encounter_id": [101, 102, 103, 104, 105],
        "patient_nbr": [1, 2, 3, 4, 5],
        "gender": ["Male", "Female", "Unknown/Invalid", "Female", "Male"],
        "discharge_disposition_id": [1, 11, 1, 19, 6],  # 11 and 19 are expired
        "readmitted": ["<30", "NO", ">30", "NO", "<30"],
        "weight": ["?", "?", "?", "?", "?"],
        "examide": ["No", "No", "No", "No", "No"],
        "citoglipton": ["No", "No", "No", "No", "No"],
    })

    cleaned = clean_and_filter_cohort(sample_data)

    # Encounters with discharge 11, 19 and gender Unknown/Invalid should be removed
    assert len(cleaned) == 2  # rows 101 and 105 remain
    assert not any(cleaned["discharge_disposition_id"].isin(EXPIRED_DISCHARGE_IDS))
    assert not any(cleaned["gender"] == "Unknown/Invalid")
    assert BINARY_TARGET_COLUMN in cleaned.columns
    assert list(cleaned[BINARY_TARGET_COLUMN]) == [1, 1]


def test_patient_grouped_split_zero_leakage():
    """Verify that no patient appears in both train and test partitions."""
    # Synthetic repeat-patient dataset
    patients = [1, 1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10]
    sample_df = pd.DataFrame({
        "encounter_id": list(range(len(patients))),
        "patient_nbr": patients,
        "readmitted_30d": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    })

    train_df, test_df = create_patient_grouped_split(sample_df, test_size=0.30, random_state=42)

    train_patients = set(train_df["patient_nbr"])
    test_patients = set(test_df["patient_nbr"])

    # Disjointness check
    overlap = train_patients.intersection(test_patients)
    assert len(overlap) == 0, f"Patient leakage detected: {overlap}"
    assert len(train_df) + len(test_df) == len(sample_df)
