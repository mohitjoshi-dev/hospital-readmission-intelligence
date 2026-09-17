"""Feature engineering, ICD-9 categorization (Strack et al. 2014), and leak-free scikit-learn transformers."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import GROUP_COLUMN, ICD9_CATEGORIES, RAW_TARGET_COLUMN, BINARY_TARGET_COLUMN

logger = logging.getLogger(__name__)


def map_icd9_to_category(code_val) -> str:
    """Map an ICD-9 diagnosis code to established disease categories from Strack et al. (2014).
    
    Standard external reference:
      Strack et al., 'Impact of HbA1c Measurement on Hospital Readmission Rates',
      BioMed Research International, 2014, Table 2.
    """
    if pd.isna(code_val):
        return "Missing"

    code_str = str(code_val).strip()
    if not code_str or code_str in ("?", "None", "NULL"):
        return "Missing"

    if code_str.startswith("V"):
        return "Supplementary_V"
    if code_str.startswith("E"):
        return "External_E"

    try:
        val = float(code_str)
    except ValueError:
        return "Other"

    int_val = int(val)

    # Circulatory: 390–459, 785
    if (390 <= int_val <= 459) or (int_val == 785):
        return "Circulatory"
    # Respiratory: 460–519, 786
    if (460 <= int_val <= 519) or (int_val == 786):
        return "Respiratory"
    # Digestive: 520–579, 787
    if (520 <= int_val <= 579) or (int_val == 787):
        return "Digestive"
    # Diabetes: 250.xx
    if int_val == 250:
        return "Diabetes"
    # Injury & Poisoning: 800–999
    if 800 <= int_val <= 999:
        return "Injury"
    # Musculoskeletal: 710–739
    if 710 <= int_val <= 739:
        return "Musculoskeletal"
    # Genitourinary: 580–629, 788
    if (580 <= int_val <= 629) or (int_val == 788):
        return "Genitourinary"
    # Neoplasms: 140–239
    if 140 <= int_val <= 239:
        return "Neoplasms"

    return "Other"


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer that engineers domain-specific healthcare features without leakage.
    
    Fitted strictly on training data; transforms both train and test.
    """

    def __init__(self):
        self.top_specialties_: List[str] = []
        self.top_payers_: List[str] = []
        self.all_med_cols: List[str] = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide",
            "glimepiride", "acetohexamide", "glipizide", "glyburide",
            "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
            "miglitol", "troglitazone", "tolazamide", "insulin",
            "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone",
            "metformin-pioglitazone",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        # Learn top 8 specialties
        if "medical_specialty" in df.columns:
            spec_counts = df["medical_specialty"].dropna().value_counts()
            self.top_specialties_ = list(spec_counts.head(8).index)

        # Learn top 6 payers
        if "payer_code" in df.columns:
            payer_counts = df["payer_code"].dropna().value_counts()
            self.top_payers_ = list(payer_counts.head(6).index)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Age Ordinal Encoding: [0-10) -> 0, [10-20) -> 1, ..., [90-100) -> 9
        age_mapping = {
            "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
            "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9
        }
        if "age" in df.columns:
            df["age_encoded"] = df["age"].map(age_mapping).fillna(5).astype(int)

        # 2. ICD-9 Categorization (Strack et al. 2014)
        for d_col in ["diag_1", "diag_2", "diag_3"]:
            if d_col in df.columns:
                df[f"{d_col}_cat"] = df[d_col].apply(map_icd9_to_category)

        # Multi-diagnosis flags
        diag_cats = [c for c in ["diag_1_cat", "diag_2_cat", "diag_3_cat"] if c in df.columns]
        if diag_cats:
            df["has_diabetes_diag"] = df[diag_cats].isin(["Diabetes"]).any(axis=1).astype(int)
            df["has_circulatory_diag"] = df[diag_cats].isin(["Circulatory"]).any(axis=1).astype(int)
            df["has_respiratory_diag"] = df[diag_cats].isin(["Respiratory"]).any(axis=1).astype(int)

        # 3. Medical Specialty & Payer Grouping
        if "medical_specialty" in df.columns:
            df["med_specialty_clean"] = df["medical_specialty"].apply(
                lambda s: s if s in self.top_specialties_ else "Other_or_Missing"
            )

        if "payer_code" in df.columns:
            df["payer_code_clean"] = df["payer_code"].apply(
                lambda p: p if p in self.top_payers_ else "Other_or_Missing"
            )

        # 4. Admission & Discharge Category Grouping
        if "admission_type_id" in df.columns:
            # 1,2,7: Urgent/Emergency/Trauma; 3: Elective; 4: Newborn; 5,6,8: Other/Not Mapped
            df["adm_type_clean"] = df["admission_type_id"].map({
                1: "Emergency", 2: "Urgent", 7: "Trauma", 3: "Elective", 4: "Newborn"
            }).fillna("Other_or_Unknown")

        if "discharge_disposition_id" in df.columns:
            # 1: Home, 6: Home Health, 3,4,5,22,23,24: Facility Transfer, 7: AMA
            def map_disp(did):
                if did in (1, 8):
                    return "Home"
                elif did == 6:
                    return "Home_Health"
                elif did in (3, 4, 5, 22, 23, 24):
                    return "Facility_Transfer"
                elif did == 7:
                    return "Left_AMA"
                else:
                    return "Other_Disposition"
            df["discharge_disp_clean"] = df["discharge_disposition_id"].apply(map_disp)

        if "admission_source_id" in df.columns:
            # 7: ER; 1,2,3: Referral; 4,5,6: Transfer
            def map_src(sid):
                if sid == 7:
                    return "Emergency_Room"
                elif sid in (1, 2, 3):
                    return "Referral"
                elif sid in (4, 5, 6, 10, 18, 22):
                    return "Transfer"
                else:
                    return "Other_Source"
            df["adm_source_clean"] = df["admission_source_id"].apply(map_src)

        # 5. Diabetes Medication Complexity & Dose Changes
        present_meds = [m for m in self.all_med_cols if m in df.columns]
        if present_meds:
            df["num_active_diabetes_meds"] = (df[present_meds] != "No").sum(axis=1)
            df["num_med_dosage_changes"] = df[present_meds].isin(["Up", "Down"]).sum(axis=1)

        # Binary treatment changes
        if "change" in df.columns:
            df["med_change_flag"] = (df["change"] == "Ch").astype(int)
        if "diabetesMed" in df.columns:
            df["diabetes_med_prescribed"] = (df["diabetesMed"] == "Yes").astype(int)

        # Key glycemic markers
        if "A1Cresult" in df.columns:
            df["has_a1c_tested"] = (df["A1Cresult"] != "None").astype(int)
            df["a1c_high"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)

        if "max_glu_serum" in df.columns:
            df["has_glucose_tested"] = (df["max_glu_serum"] != "None").astype(int)
            df["glucose_high"] = df["max_glu_serum"].isin([">200", ">300"]).astype(int)

        # 6. Prior Utilization & Velocity
        num_cols = ["number_inpatient", "number_emergency", "number_outpatient"]
        if all(c in df.columns for c in num_cols):
            df["total_prior_visits"] = df["number_inpatient"] + df["number_emergency"] + df["number_outpatient"]
            df["prior_inpatient_ratio"] = df["number_inpatient"] / (df["total_prior_visits"] + 1)
            df["high_acute_utilizer"] = (
                (df["number_inpatient"] >= 2) | (df["number_emergency"] >= 2)
            ).astype(int)

        return df


def build_preprocessor_pipeline() -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Construct the scikit-learn ColumnTransformer for standardizing and encoding features."""
    numeric_features = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "age_encoded",
        "num_active_diabetes_meds",
        "num_med_dosage_changes",
        "total_prior_visits",
        "prior_inpatient_ratio",
        "has_diabetes_diag",
        "has_circulatory_diag",
        "has_respiratory_diag",
        "med_change_flag",
        "diabetes_med_prescribed",
        "has_a1c_tested",
        "a1c_high",
        "has_glucose_tested",
        "glucose_high",
        "high_acute_utilizer",
    ]

    categorical_features = [
        "race",
        "gender",
        "diag_1_cat",
        "diag_2_cat",
        "diag_3_cat",
        "adm_type_clean",
        "discharge_disp_clean",
        "adm_source_clean",
        "med_specialty_clean",
        "payer_code_clean",
        "insulin",
        "metformin",
        "A1Cresult",
        "max_glu_serum",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features
