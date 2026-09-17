"""Model training, internal patient-grouped cross-validation, and probability calibration."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.config import (
    BINARY_TARGET_COLUMN,
    CV_FOLDS,
    GROUP_COLUMN,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.feature_pipeline import ClinicalFeatureEngineer, build_preprocessor_pipeline

logger = logging.getLogger(__name__)


def build_candidate_models(random_state: int = RANDOM_SEED) -> Dict[str, Any]:
    """Define the lean model candidates specified by the operational plan."""
    return {
        "Majority_Class_Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic_Regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def cross_validate_on_train_set(
    train_df: pd.DataFrame,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Execute patient-grouped cross-validation strictly inside the 80% training set.
    
    Guarantees no patient appears in both fold-train and fold-val, and test set remains untouched.
    """
    if GROUP_COLUMN not in train_df.columns:
        raise KeyError(f"Group column '{GROUP_COLUMN}' missing from train_df.")

    groups = train_df[GROUP_COLUMN].values
    gkf = GroupKFold(n_splits=n_splits)

    candidate_models = build_candidate_models(random_state=random_state)
    results: Dict[str, Dict[str, List[float]]] = {
        name: {"roc_auc": [], "pr_auc": [], "brier": []} for name in candidate_models
    }

    logger.info("Initiating %d-fold patient-grouped cross-validation strictly within training set...", n_splits)

    fold = 1
    for train_idx, val_idx in gkf.split(train_df, groups=groups):
        fold_train_df = train_df.iloc[train_idx].copy()
        fold_val_df = train_df.iloc[val_idx].copy()

        y_train = fold_train_df[BINARY_TARGET_COLUMN].values
        y_val = fold_val_df[BINARY_TARGET_COLUMN].values

        # 1. Feature Engineering (fitted strictly on fold_train)
        feat_eng = ClinicalFeatureEngineer()
        feat_eng.fit(fold_train_df)
        X_train_fe = feat_eng.transform(fold_train_df)
        X_val_fe = feat_eng.transform(fold_val_df)

        # 2. Preprocessor (fitted strictly on fold_train)
        preprocessor, _, _ = build_preprocessor_pipeline()
        X_train_trans = preprocessor.fit_transform(X_train_fe)
        X_val_trans = preprocessor.transform(X_val_fe)

        for name, model_cls in candidate_models.items():
            # Clone model for fold
            model = build_candidate_models(random_state=random_state)[name]
            model.fit(X_train_trans, y_train)

            # Predict probabilities
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val_trans)[:, 1]
            else:
                y_prob = np.full(len(y_val), np.mean(y_train))

            roc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.50
            pr = average_precision_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.0
            brier = brier_score_loss(y_val, y_prob)

            results[name]["roc_auc"].append(roc)
            results[name]["pr_auc"].append(pr)
            results[name]["brier"].append(brier)

        logger.info("Completed Fold %d/%d", fold, n_splits)
        fold += 1

    summary_rows = []
    for name, metric_dict in results.items():
        summary_rows.append({
            "Model": name,
            "CV ROC-AUC (Mean)": np.round(np.mean(metric_dict["roc_auc"]), 4),
            "CV ROC-AUC (Std)": np.round(np.std(metric_dict["roc_auc"]), 4),
            "CV PR-AUC (Mean)": np.round(np.mean(metric_dict["pr_auc"]), 4),
            "CV PR-AUC (Std)": np.round(np.std(metric_dict["pr_auc"]), 4),
            "CV Brier (Mean)": np.round(np.mean(metric_dict["brier"]), 4),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="CV PR-AUC (Mean)", ascending=False).reset_index(drop=True)
    logger.info("Internal Training Cross-Validation Summary:\n%s", summary_df.to_string())
    return summary_df


def train_and_calibrate_pipeline(
    train_df: pd.DataFrame,
    model_name: str = "Random_Forest",
    random_state: int = RANDOM_SEED,
) -> Tuple[Any, ClinicalFeatureEngineer, Any, List[str]]:
    """Fit full feature pipeline and probability-calibrated model on training cohort.
    
    Uses a patient-grouped internal split inside train_df to fit probability calibration
    without ever touching the held-out 20% test partition.
    """
    logger.info("Fitting and calibrating %s on full training set...", model_name)

    # Internal split for calibration
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    groups = train_df[GROUP_COLUMN].values
    fit_idx, calib_idx = next(gss.split(train_df, groups=groups))

    fit_df = train_df.iloc[fit_idx].copy()
    calib_df = train_df.iloc[calib_idx].copy()

    # 1. Fit Feature Engineer on train_df
    feat_eng = ClinicalFeatureEngineer()
    feat_eng.fit(train_df)

    fit_fe = feat_eng.transform(fit_df)
    calib_fe = feat_eng.transform(calib_df)

    # 2. Fit Preprocessor on fit_fe
    preprocessor, num_cols, cat_cols = build_preprocessor_pipeline()
    X_fit = preprocessor.fit_transform(fit_fe)
    X_calib = preprocessor.transform(calib_fe)

    y_fit = fit_df[BINARY_TARGET_COLUMN].values
    y_calib = calib_df[BINARY_TARGET_COLUMN].values

    # 3. Fit Base Classifier
    models = build_candidate_models(random_state=random_state)
    base_model = models.get(model_name, models["Random_Forest"])
    base_model.fit(X_fit, y_fit)

    # 4. Calibrate Probabilities (Sigmoid / Platt scaling on internal calibration partition)
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method="sigmoid")
    except (ImportError, TypeError):
        calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")

    calibrated_model.fit(X_calib, y_calib)

    # Get feature names for interpretability
    try:
        cat_encoder = preprocessor.named_transformers_["cat"]
        encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        feature_names = num_cols + encoded_cat_names
    except Exception:
        feature_names = [f"feat_{i}" for i in range(X_fit.shape[1])]

    logger.info("Successfully calibrated %s pipeline. Total features: %d", model_name, len(feature_names))
    return calibrated_model, feat_eng, preprocessor, feature_names


def save_artifacts(
    calibrated_model: Any,
    feat_eng: ClinicalFeatureEngineer,
    preprocessor: Any,
    feature_names: List[str],
    prefix: str = "readmission_model",
) -> None:
    """Save trained pipeline artifacts to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, MODELS_DIR / f"{prefix}_calibrated.joblib")
    joblib.dump(feat_eng, MODELS_DIR / f"{prefix}_feat_eng.joblib")
    joblib.dump(preprocessor, MODELS_DIR / f"{prefix}_preprocessor.joblib")
    joblib.dump(feature_names, MODELS_DIR / f"{prefix}_feature_names.joblib")
    logger.info("Saved model artifacts to %s", MODELS_DIR)


def load_artifacts(
    prefix: str = "readmission_model",
) -> Tuple[Any, ClinicalFeatureEngineer, Any, List[str]]:
    """Load persisted model artifacts from disk."""
    calibrated_model = joblib.load(MODELS_DIR / f"{prefix}_calibrated.joblib")
    feat_eng = joblib.load(MODELS_DIR / f"{prefix}_feat_eng.joblib")
    preprocessor = joblib.load(MODELS_DIR / f"{prefix}_preprocessor.joblib")
    feature_names = joblib.load(MODELS_DIR / f"{prefix}_feature_names.joblib")
    return calibrated_model, feat_eng, preprocessor, feature_names
