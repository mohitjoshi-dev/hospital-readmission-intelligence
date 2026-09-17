"""Full training, cross-validation, test set evaluation, and artifact generation pipeline."""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BINARY_TARGET_COLUMN,
    CLEANED_DATASET_FILE,
    DEFAULT_CAPACITY_PRESETS,
    GROUP_COLUMN,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_DATA_FILE,
    TRAIN_DATA_FILE,
)
from src.data_loader import clean_and_filter_cohort, create_patient_grouped_split, load_raw_data
from src.evaluation import compute_comprehensive_metrics, compute_fairness_monitoring, get_curve_data
from src.explanation import OperationalExplainer
from src.feature_pipeline import ClinicalFeatureEngineer, build_preprocessor_pipeline
from src.modeling import (
    build_candidate_models,
    cross_validate_on_train_set,
    save_artifacts,
    train_and_calibrate_pipeline,
)
from src.operational_queue import generate_operational_review_queue

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_pipeline() -> None:
    """Execute end-to-end data processing, training, calibration, and test evaluation."""
    logger.info("=== STEP 1: Ingestion and Cohort Filtering ===")
    raw_df = load_raw_data()
    cleaned_df = clean_and_filter_cohort(raw_df)
    cleaned_df.to_csv(CLEANED_DATASET_FILE, index=False)
    logger.info("Saved cleaned cohort to: %s", CLEANED_DATASET_FILE)

    logger.info("=== STEP 2: Patient-Grouped 80/20 Train/Test Splitting ===")
    train_df, test_df = create_patient_grouped_split(cleaned_df, test_size=0.20, random_state=RANDOM_SEED)
    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)
    logger.info("Saved train (%d rows) and test (%d rows) splits to disk.", len(train_df), len(test_df))

    logger.info("=== STEP 3: Internal Training Cross-Validation (5-Fold GroupKFold) ===")
    cv_summary_df = cross_validate_on_train_set(train_df, n_splits=5, random_state=RANDOM_SEED)
    cv_summary_df.to_csv(REPORTS_DIR / "cv_model_comparison.csv", index=False)

    logger.info("=== STEP 4: Full Training & Probability Calibration ===")
    calibrated_model, feat_eng, preprocessor, feature_names = train_and_calibrate_pipeline(
        train_df=train_df,
        model_name="Random_Forest",
        random_state=RANDOM_SEED,
    )
    save_artifacts(calibrated_model, feat_eng, preprocessor, feature_names)

    logger.info("=== STEP 5: Held-Out 20% Test Set Evaluation ===")
    # Preprocess test set using pipelines fitted strictly on training data
    test_fe = feat_eng.transform(test_df)
    X_test = preprocessor.transform(test_fe)
    y_test = test_df[BINARY_TARGET_COLUMN].values

    # Evaluate Candidate Models on Test Set
    test_results = []
    trained_candidates = {}

    # Fit baselines on train for test benchmarking
    train_fe = feat_eng.transform(train_df)
    X_train = preprocessor.transform(train_fe)
    y_train = train_df[BINARY_TARGET_COLUMN].values

    models = build_candidate_models(random_state=RANDOM_SEED)
    for model_name, model in models.items():
        logger.info("Fitting and evaluating %s on test set...", model_name)
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            test_probs = model.predict_proba(X_test)[:, 1]
        else:
            test_probs = np.full(len(y_test), np.mean(y_train))

        trained_candidates[model_name] = model
        metrics = compute_comprehensive_metrics(y_test, test_probs, threshold=0.50)
        metrics["model_name"] = model_name
        test_results.append(metrics)

    # Calibrated Random Forest on Test Set
    calib_probs = calibrated_model.predict_proba(X_test)[:, 1]
    calib_metrics = compute_comprehensive_metrics(y_test, calib_probs, threshold=0.50)
    calib_metrics["model_name"] = "Calibrated_Random_Forest"
    test_results.append(calib_metrics)

    test_metrics_df = pd.DataFrame(test_results)
    test_metrics_df.to_csv(REPORTS_DIR / "test_model_benchmarks.csv", index=False)
    logger.info("Test Set Benchmarks:\n%s", test_metrics_df[[
        "model_name", "roc_auc", "pr_auc", "f1_score", "brier_score"
    ]].to_string())

    logger.info("=== STEP 6: Curve Data Generation ===")
    curve_data = get_curve_data(y_test, calib_probs)
    # Convert numpy arrays to serializable lists
    curve_json = {
        "fpr": [float(x) for x in curve_data["fpr"]],
        "tpr": [float(x) for x in curve_data["tpr"]],
        "roc_auc": float(curve_data["roc_auc"]),
        "precision": [float(x) for x in curve_data["precision"]],
        "recall": [float(x) for x in curve_data["recall"]],
        "pr_auc": float(curve_data["pr_auc"]),
        "prob_true": [float(x) for x in curve_data["prob_true"]],
        "prob_pred": [float(x) for x in curve_data["prob_pred"]],
        "brier_score": float(curve_data["brier_score"]),
    }
    with open(REPORTS_DIR / "curve_data.json", "w", encoding="utf-8") as f:
        json.dump(curve_json, f, indent=2)

    logger.info("=== STEP 7: Operational Capacity Queues & Lift Analysis ===")
    capacity_reports = {}
    for preset_name, ratio in DEFAULT_CAPACITY_PRESETS.items():
        queue_df, cap_metrics = generate_operational_review_queue(
            df=test_fe,
            predicted_probabilities=calib_probs,
            capacity_ratio=ratio,
            actual_labels=y_test,
        )
        capacity_reports[preset_name] = cap_metrics

    with open(REPORTS_DIR / "capacity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(capacity_reports, f, indent=2)
    logger.info("Operational Capacity Analysis:\n%s", json.dumps(capacity_reports, indent=2))

    # Save test predictions for dashboard
    test_export = test_df[[
        GROUP_COLUMN, "age", "race", "gender", "time_in_hospital",
        "num_medications", "number_inpatient", "number_emergency", "number_diagnoses",
        "admission_type_id", "discharge_disposition_id"
    ]].copy()
    test_export["actual_readmitted_30d"] = y_test
    test_export["predicted_risk_pct"] = np.round(calib_probs * 100, 2)
    test_export["predicted_probability"] = calib_probs

    # Assign default 10% operational queue
    queue_10_df, _ = generate_operational_review_queue(
        df=test_export,
        predicted_probabilities=calib_probs,
        capacity_ratio=0.10,
        actual_labels=y_test,
    )
    queue_10_df.to_csv(REPORTS_DIR / "test_operational_queue.csv", index=False)

    logger.info("=== STEP 8: Demographic Fairness Monitoring Audit ===")
    test_eval_df = test_export.copy()
    op_cutoff = float(capacity_reports["Top 10%"]["cutoff_probability"])
    fairness_results = compute_fairness_monitoring(
        df=test_eval_df,
        y_true_col="actual_readmitted_30d",
        y_pred_proba_col="predicted_probability",
        threshold=op_cutoff,
    )
    for sensitive_col, f_table in fairness_results.items():
        f_table.to_csv(REPORTS_DIR / f"fairness_{sensitive_col}.csv", index=False)
        logger.info("Fairness Monitoring [%s]:\n%s", sensitive_col, f_table.to_string())

    logger.info("=== STEP 9: SHAP Global Operational Attribution ===")
    try:
        explainer = OperationalExplainer(calibrated_model, feature_names)
        # Sample 400 records for fast, stable SHAP estimation
        sample_indices = np.random.RandomState(RANDOM_SEED).choice(len(X_test), size=min(400, len(X_test)), replace=False)
        X_sample = X_test[sample_indices]
        global_importance = explainer.get_global_importance(X_sample, top_n=20)
        global_importance.to_csv(REPORTS_DIR / "global_shap_importance.csv", index=False)
        logger.info("Global SHAP Drivers:\n%s", global_importance.to_string())
    except Exception as e:
        logger.warning("SHAP computation note: %s", e)

    logger.info("=== PIPELINE EXECUTION SUCCESSFULLY COMPLETED ===")


if __name__ == "__main__":
    run_pipeline()
