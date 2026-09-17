"""Evaluation metrics, threshold analysis, ROC/PR curves, Brier calibration, and demographic fairness monitoring."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Union[float, int]]:
    """Compute standard classification metrics including ROC-AUC, PR-AUC, F1, and Brier score."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.50
    pr_auc = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
    brier = brier_score_loss(y_true, y_pred_proba)

    metrics = {
        "threshold": float(np.round(threshold, 4)),
        "accuracy": float(np.round(acc, 4)),
        "precision": float(np.round(prec, 4)),
        "recall": float(np.round(rec, 4)),
        "f1_score": float(np.round(f1, 4)),
        "roc_auc": float(np.round(roc_auc, 4)),
        "pr_auc": float(np.round(pr_auc, 4)),
        "brier_score": float(np.round(brier, 4)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_samples": int(len(y_true)),
        "base_rate": float(np.round(np.mean(y_true) * 100, 2)),
    }

    return metrics


def compute_fairness_monitoring(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_proba_col: str,
    threshold: float = 0.50,
    sensitive_columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Audit demographic sub-cohorts for administrative fairness and operational disparity monitoring.
    
    Monitors Base Rate, Selection Rate, False Positive Rate (FPR), and Recall (TPR)
    across race, gender, and age categories to detect disparate impact.
    Strictly an operational monitoring component; not a clinical rule.
    """
    if sensitive_columns is None:
        sensitive_columns = ["race", "gender", "age"]

    fairness_tables: Dict[str, pd.DataFrame] = {}

    df_eval = df.copy()
    df_eval["y_pred"] = (df_eval[y_pred_proba_col] >= threshold).astype(int)

    # Simplified age groupings for cleaner monitoring
    if "age" in df_eval.columns:
        def simplify_age(val):
            if str(val) in ("[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)"):
                return "< 50 years"
            elif str(val) in ("[50-60)", "[60-70)"):
                return "50 - 70 years"
            elif str(val) in ("[70-80)", "[80-90)", "[90-100)"):
                return "> 70 years"
            return "Unknown"
        df_eval["age_cohort"] = df_eval["age"].apply(simplify_age)
        sensitive_cols_effective = [c if c != "age" else "age_cohort" for c in sensitive_columns]
    else:
        sensitive_cols_effective = sensitive_columns

    for col in sensitive_cols_effective:
        if col not in df_eval.columns:
            continue

        rows = []
        for group_val, group_sub in df_eval.groupby(col):
            if pd.isna(group_val) or group_val in ("?", "Unknown/Invalid"):
                group_name = "Missing/Unknown"
            else:
                group_name = str(group_val)

            n_samples = len(group_sub)
            if n_samples < 20:
                continue

            y_sub_true = group_sub[y_true_col].values
            y_sub_pred = group_sub["y_pred"].values
            y_sub_prob = group_sub[y_pred_proba_col].values

            tn, fp, fn, tp = confusion_matrix(y_sub_true, y_sub_pred, labels=[0, 1]).ravel()

            base_rate = float(np.mean(y_sub_true) * 100)
            selection_rate = float(np.mean(y_sub_pred) * 100)
            recall = float((tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0)
            fpr = float((fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0)
            precision = float((tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0)
            roc_auc = float(roc_auc_score(y_sub_true, y_sub_prob)) if len(np.unique(y_sub_true)) > 1 else 0.50

            rows.append({
                "Demographic Group": group_name,
                "Sample Count": n_samples,
                "Base Readmit Rate (%)": np.round(base_rate, 2),
                "Selection Rate (%)": np.round(selection_rate, 2),
                "Recall / TPR (%)": np.round(recall, 2),
                "False Positive Rate (%)": np.round(fpr, 2),
                "Precision (%)": np.round(precision, 2),
                "ROC-AUC": np.round(roc_auc, 3),
            })

        table_df = pd.DataFrame(rows)
        if not table_df.empty:
            table_df = table_df.sort_values(by="Sample Count", ascending=False).reset_index(drop=True)
            fairness_tables[col] = table_df

    return fairness_tables


def get_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, Union[np.ndarray, float]]:
    """Generate ROC, Precision-Recall, and Calibration reliability points."""
    fpr, tpr, roc_thresh = roc_curve(y_true, y_pred_proba)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_pred_proba)
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10, strategy="uniform")

    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    brier = float(brier_score_loss(y_true, y_pred_proba))

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresh,
        "roc_auc": roc_auc,
        "precision": prec,
        "recall": rec,
        "pr_thresholds": pr_thresh,
        "pr_auc": pr_auc,
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "brier_score": brier,
    }
