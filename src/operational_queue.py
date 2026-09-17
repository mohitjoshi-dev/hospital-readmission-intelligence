"""Operational review queue engine and capacity-based prioritization workflows."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_CAPACITY_PRESETS,
    WORKFLOW_ADDITIONAL_RECORD,
    WORKFLOW_CARE_MANAGEMENT,
    WORKFLOW_DISCHARGE_PLANNING,
    WORKFLOW_FOLLOWUP_COORDINATION,
)

logger = logging.getLogger(__name__)


def assign_operational_workflows(row: pd.Series) -> List[str]:
    """Assign administrative review workflows based on operational indicators.
    
    Strictly non-clinical administrative review routing:
      - Care-Management Review: acute past utilization (>=2 inpatient or emergency visits)
      - Discharge-Planning Review: non-routine discharge disposition (SNF, Home Health, AMA)
      - Follow-Up Coordination Review: active diabetes medication regimen adjustments
      - Additional Record Review: high diagnostic complexity (>=9 diagnoses or multi-comorbidity)
    """
    workflows = []

    # 1. Care-Management Review
    num_inp = row.get("number_inpatient", 0)
    num_er = row.get("number_emergency", 0)
    if num_inp >= 2 or num_er >= 2:
        workflows.append(WORKFLOW_CARE_MANAGEMENT)

    # 2. Discharge-Planning Review
    disp_clean = row.get("discharge_disp_clean", "")
    raw_disp = row.get("discharge_disposition_id", 1)
    if disp_clean in ("Facility_Transfer", "Home_Health", "Left_AMA") or raw_disp in (3, 4, 5, 6, 7, 22, 23, 24):
        workflows.append(WORKFLOW_DISCHARGE_PLANNING)

    # 3. Follow-Up Coordination Review
    med_change = row.get("med_change_flag", 0)
    raw_change = row.get("change", "No")
    insulin_val = str(row.get("insulin", "No"))
    if med_change == 1 or raw_change == "Ch" or insulin_val in ("Up", "Down"):
        workflows.append(WORKFLOW_FOLLOWUP_COORDINATION)

    # 4. Additional Record Review
    num_diag = row.get("number_diagnoses", 0)
    has_circ = row.get("has_circulatory_diag", 0)
    has_resp = row.get("has_respiratory_diag", 0)
    if num_diag >= 9 or (has_circ == 1 and has_resp == 1):
        workflows.append(WORKFLOW_ADDITIONAL_RECORD)

    if not workflows:
        workflows.append("Standard Administrative Review")

    return workflows


def generate_operational_review_queue(
    df: pd.DataFrame,
    predicted_probabilities: np.ndarray,
    capacity_ratio: float = 0.10,
    actual_labels: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
    """Generate a capacity-ranked operational review queue sorted by predicted risk.
    
    Args:
        df: Input DataFrame containing patient encounter features.
        predicted_probabilities: Calibrated probability of 30-day readmission [0.0, 1.0].
        capacity_ratio: Fraction of total patient population hospital staff can review (e.g. 0.05, 0.10, 0.20).
        actual_labels: Optional true binary target for computing capture rate (lift).
        
    Returns:
        Tuple of (ranked_queue_df, capacity_summary_metrics).
    """
    if len(df) != len(predicted_probabilities):
        raise ValueError(
            f"Row count mismatch: df has {len(df)} rows, probabilities has {len(predicted_probabilities)}."
        )

    queue_df = df.copy()
    queue_df["predicted_risk_pct"] = np.round(predicted_probabilities * 100, 2)
    queue_df["predicted_probability"] = predicted_probabilities

    if actual_labels is not None:
        queue_df["actual_readmitted_30d"] = actual_labels

    # Sort descending by predicted readmission probability
    queue_df = queue_df.sort_values(by="predicted_probability", ascending=False).reset_index(drop=True)
    queue_df["queue_rank"] = np.arange(1, len(queue_df) + 1)

    # Determine capacity cutoff
    total_encounters = len(queue_df)
    cutoff_count = max(1, int(np.ceil(total_encounters * capacity_ratio)))
    cutoff_prob = queue_df.iloc[cutoff_count - 1]["predicted_probability"]

    queue_df["operational_tier"] = np.where(
        queue_df["queue_rank"] <= cutoff_count,
        "PRIORITIZED OPERATIONAL QUEUE",
        "Standard Administrative Workflow",
    )

    # Assign operational workflows
    queue_df["assigned_workflows"] = queue_df.apply(assign_operational_workflows, axis=1)
    queue_df["assigned_workflows_str"] = queue_df["assigned_workflows"].apply(lambda wfs: "; ".join(wfs))

    # Metrics summary
    prioritized_df = queue_df.iloc[:cutoff_count]
    metrics: Dict[str, Union[int, float]] = {
        "total_encounters": total_encounters,
        "capacity_ratio": capacity_ratio,
        "capacity_percentage": np.round(capacity_ratio * 100, 1),
        "prioritized_count": cutoff_count,
        "cutoff_probability": float(np.round(cutoff_prob, 4)),
        "cutoff_risk_pct": float(np.round(cutoff_prob * 100, 2)),
        "average_risk_in_prioritized_queue": float(np.round(prioritized_df["predicted_risk_pct"].mean(), 2)),
    }

    if actual_labels is not None:
        total_actual_readmissions = int(np.sum(actual_labels))
        captured_readmissions = int(prioritized_df["actual_readmitted_30d"].sum())
        capture_rate = (captured_readmissions / total_actual_readmissions * 100) if total_actual_readmissions > 0 else 0.0
        precision_at_k = (captured_readmissions / cutoff_count * 100) if cutoff_count > 0 else 0.0

        metrics["total_actual_readmissions"] = total_actual_readmissions
        metrics["captured_readmissions"] = captured_readmissions
        metrics["capture_rate_pct"] = float(np.round(capture_rate, 2))
        metrics["precision_at_capacity_pct"] = float(np.round(precision_at_k, 2))
        metrics["lift"] = float(np.round(capture_rate / (capacity_ratio * 100), 2)) if capacity_ratio > 0 else 1.0

    logger.info(
        "Generated operational queue: %d encounters prioritized (Capacity: %.1f%%, Cutoff: %.2f%%)",
        cutoff_count,
        capacity_ratio * 100,
        cutoff_prob * 100,
    )

    return queue_df, metrics
