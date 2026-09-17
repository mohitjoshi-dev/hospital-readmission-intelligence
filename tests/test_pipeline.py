"""Automated unit tests for ICD-9 category mappings, operational queue cutoffs, and workflow assignment."""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    WORKFLOW_ADDITIONAL_RECORD,
    WORKFLOW_CARE_MANAGEMENT,
    WORKFLOW_DISCHARGE_PLANNING,
    WORKFLOW_FOLLOWUP_COORDINATION,
)
from src.feature_pipeline import map_icd9_to_category
from src.operational_queue import assign_operational_workflows, generate_operational_review_queue


def test_strack_icd9_groupings():
    """Verify published Strack et al. (2014) ICD-9 disease category mapping rules."""
    test_cases = [
        ("414.01", "Circulatory"),
        ("428", "Circulatory"),
        ("785", "Circulatory"),
        ("486", "Respiratory"),
        ("519.8", "Respiratory"),
        ("530.81", "Digestive"),
        ("250.02", "Diabetes"),
        ("250.8", "Diabetes"),
        ("820.2", "Injury"),
        ("715.9", "Musculoskeletal"),
        ("599.0", "Genitourinary"),
        ("174.9", "Neoplasms"),
        ("V58.61", "Supplementary_V"),
        ("E876.0", "External_E"),
        ("?", "Missing"),
        (None, "Missing"),
        ("9999", "Other"),
    ]

    for code, expected_cat in test_cases:
        assert map_icd9_to_category(code) == expected_cat, f"Failed for code: {code}"


def test_operational_queue_capacity_cutoff():
    """Verify that operational capacity ranking precisely cuts off at specified capacity ratio."""
    n_records = 100
    df = pd.DataFrame({
        "patient_nbr": list(range(n_records)),
        "number_inpatient": [0] * n_records,
        "number_emergency": [0] * n_records,
    })
    # Simulated probabilities from 0.99 down to 0.00
    probs = np.linspace(0.99, 0.01, n_records)
    actuals = np.array([1 if p > 0.8 else 0 for p in probs])

    # Test 10% capacity
    queue_10, metrics_10 = generate_operational_review_queue(
        df=df,
        predicted_probabilities=probs,
        capacity_ratio=0.10,
        actual_labels=actuals,
    )

    assert metrics_10["prioritized_count"] == 10
    assert (queue_10["operational_tier"] == "PRIORITIZED OPERATIONAL QUEUE").sum() == 10
    assert queue_10.iloc[0]["queue_rank"] == 1
    assert queue_10.iloc[0]["predicted_risk_pct"] == 99.0

    # Test 25% capacity
    queue_25, metrics_25 = generate_operational_review_queue(
        df=df,
        predicted_probabilities=probs,
        capacity_ratio=0.25,
        actual_labels=actuals,
    )
    assert metrics_25["prioritized_count"] == 25


def test_operational_workflow_assignment():
    """Verify that operational review tags are correctly assigned based on administrative flags."""
    # Acute utilization patient -> Care-Management Review
    row_cm = pd.Series({"number_inpatient": 2, "number_emergency": 1})
    assert WORKFLOW_CARE_MANAGEMENT in assign_operational_workflows(row_cm)

    # Complex discharge disposition -> Discharge-Planning Review
    row_dp = pd.Series({"discharge_disp_clean": "Facility_Transfer"})
    assert WORKFLOW_DISCHARGE_PLANNING in assign_operational_workflows(row_dp)

    # Medication change -> Follow-Up Coordination Review
    row_fu = pd.Series({"med_change_flag": 1, "insulin": "Up"})
    assert WORKFLOW_FOLLOWUP_COORDINATION in assign_operational_workflows(row_fu)

    # Diagnostic complexity -> Additional Record Review
    row_ar = pd.Series({"number_diagnoses": 10, "has_circulatory_diag": 1, "has_respiratory_diag": 1})
    assert WORKFLOW_ADDITIONAL_RECORD in assign_operational_workflows(row_ar)
