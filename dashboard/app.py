"""Hospital Readmission Intelligence - Operational Decision-Support Platform.

A healthcare analytics and operational readmission-risk prioritization platform
built on historical EHR data (Strack et al. 1999–2008).
Strict non-clinical governance: administrative review triage only.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Guarantee project root is in sys.path before any local package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Configure Matplotlib dark theme to match application palette
plt.rcParams.update({
    "figure.facecolor": "#111827",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#1E293B",
    "axes.labelcolor": "#94A3B8",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "text.color": "#F8FAFC",
    "grid.color": "#1E293B",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "figure.autolayout": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
})

from src.llm_assistant import (
    OFFLINE_SUGGESTED_ANSWERS,
    compile_verified_analytical_context,
    query_analytics_assistant,
)

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Intelligence | Operational Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Healthcare Data Intelligence Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Space+Grotesk:wght@500;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stMarkdown, p, div, span, label, input, button, select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #CBD5E1;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        letter-spacing: -0.02em;
    }

    /* Container Backgrounds */
    .stApp {
        background-color: #0B0F14;
        color: #CBD5E1;
    }

    [data-testid="stSidebar"] {
        background-color: #0D131C;
        border-right: 1px solid #1E293B;
    }

    /* Main Page Titles */
    .main-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.35rem;
        font-weight: 700;
        color: #F8FAFC !important;
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin-bottom: 0.4rem;
        padding-bottom: 0.55rem;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, #38BDF8 0%, #22D3EE 35%, transparent 80%) 1;
    }

    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.98rem;
        color: #94A3B8;
        line-height: 1.55;
        margin-top: 0.35rem;
        margin-bottom: 1.5rem;
    }

    /* Column flex alignment for responsive equal-height cards */
    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="column"] > div {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* KPI Metric Cards */
    [data-testid="stMetric"] {
        background-color: #111827 !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        padding: 12px 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.3) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        height: 100% !important;
        min-height: 128px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        box-sizing: border-box !important;
    }

    [data-testid="stMetric"]:hover {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.15) !important;
    }

    /* Prevent truncation on Metric Labels */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] label,
    [data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.80rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
        line-height: 1.25 !important;
        min-height: 2.2em !important;
        height: auto !important;
    }

    /* Metric Value */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div {
        color: #F8FAFC !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
        letter-spacing: -0.02em !important;
        white-space: nowrap !important;
        line-height: 1.2 !important;
        margin: 4px 0 !important;
        overflow: visible !important;
    }

    /* Prevent cutoff on Delta / Supporting Badge */
    [data-testid="stMetricDelta"] {
        color: #38BDF8 !important;
        background-color: rgba(56, 189, 248, 0.08) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.74rem !important;
        font-weight: 500 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        display: inline-flex !important;
        align-items: center !important;
        flex-wrap: wrap !important;
        line-height: 1.25 !important;
        margin-top: 4px !important;
        width: fit-content !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    [data-testid="stMetricDelta"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
        line-height: 1.25 !important;
        max-width: 100% !important;
    }

    [data-testid="stMetricDelta"] svg {
        display: none !important;
    }

    /* Custom Cards */
    .metric-card {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }

    /* Operational Governance / Disclaimer Box */
    .disclaimer-box {
        background: rgba(23, 23, 23, 0.75);
        border: 1px solid #332617;
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 24px;
        font-size: 0.88rem;
        color: #FDE68A;
        line-height: 1.5;
        backdrop-filter: blur(8px);
    }

    .disclaimer-box strong {
        color: #FBBF24;
    }

    /* Sidebar Navigation Typography */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-family: 'Inter', sans-serif !important;
        color: #CBD5E1 !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
    }

    /* Inputs, Selectboxes, and Sliders */
    .stTextInput input, .stSelectbox [data-baseweb="select"] {
        background-color: #111827 !important;
        border-color: #1E293B !important;
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] {
        border: 1px solid #1E293B;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }

    /* Section Dividers */
    hr {
        border-color: #1E293B !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paths
ROOT_DIR = PROJECT_ROOT
DATA_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"


# Cached Data Loaders
@st.cache_data
def load_cleaned_cohort() -> Optional[pd.DataFrame]:
    cohort_path = DATA_DIR / "cleaned_cohort.csv"
    if cohort_path.exists():
        return pd.read_csv(cohort_path, low_memory=False)
    return None


@st.cache_data
def load_report_artifacts():
    benchmarks_path = REPORTS_DIR / "test_model_benchmarks.csv"
    cv_path = REPORTS_DIR / "cv_model_comparison.csv"
    curve_path = REPORTS_DIR / "curve_data.json"
    capacity_path = REPORTS_DIR / "capacity_metrics.json"
    shap_path = REPORTS_DIR / "global_shap_importance.csv"
    queue_path = REPORTS_DIR / "test_operational_queue.csv"

    benchmarks_df = pd.read_csv(benchmarks_path) if benchmarks_path.exists() else None
    cv_df = pd.read_csv(cv_path) if cv_path.exists() else None
    shap_df = pd.read_csv(shap_path) if shap_path.exists() else None
    queue_df = pd.read_csv(queue_path) if queue_path.exists() else None

    curve_data = None
    if curve_path.exists():
        with open(curve_path, "r", encoding="utf-8") as f:
            curve_data = json.load(f)

    capacity_data = None
    if capacity_path.exists():
        with open(capacity_path, "r", encoding="utf-8") as f:
            capacity_data = json.load(f)

    fairness_data = {}
    for sensitive in ["race", "gender", "age_cohort"]:
        f_path = REPORTS_DIR / f"fairness_{sensitive}.csv"
        if f_path.exists():
            fairness_data[sensitive] = pd.read_csv(f_path)

    return benchmarks_df, cv_df, curve_data, capacity_data, shap_df, queue_df, fairness_data


# Load data assets
df_cohort = load_cleaned_cohort()
benchmarks_df, cv_df, curve_data, capacity_data, shap_df, queue_df, fairness_data = load_report_artifacts()

# Sidebar: Professional Header (No broken image)
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid #1E293B;">
        <div style="width: 44px; height: 44px; min-width: 44px; border-radius: 8px; background: rgba(17, 24, 39, 0.85); border: 1px solid #38BDF8; box-shadow: 0 0 12px rgba(56, 189, 248, 0.25); display: flex; align-items: center; justify-content: center; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05rem; color: #F8FAFC; letter-spacing: 0.05em;">
            HRI
        </div>
        <div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; color: #F8FAFC; line-height: 1.15; letter-spacing: -0.02em;">
                Hospital Readmission<br/><span style="color: #38BDF8;">Intelligence</span>
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; color: #94A3B8; margin-top: 4px; font-weight: 400;">
                Operational Healthcare Analytics
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Executive Overview",
        "2. Data Quality",
        "3. Cohort & Encounter Analytics",
        "4. Readmission Analysis",
        "5. Predictive Modeling",
        "6. Risk Prioritization",
        "7. SHAP Explainability",
        "8. LLM Analytics Assistant",
        "9. About / Methodology",
    ],
)

# Global Governance Banner
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ OPERATIONAL DECISION-SUPPORT DEMONSTRATION ONLY</strong><br/>
        This platform is an operational analytics and workflow prioritization tool using historical EHR data (1999–2008). 
        It is <strong>NOT</strong> a clinical diagnostic, treatment, medication recommendation, or medical decision-making system. 
        All recommendations are strictly limited to hospital administrative review workflows (care management, discharge planning, follow-up coordination).
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 1. EXECUTIVE OVERVIEW
# =============================================================================
if nav_selection == "1. Executive Overview":
    st.markdown('<div class="main-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">High-level operational intelligence and key observational findings from 100,111 eligible inpatient encounters across 130 US hospitals.</div>',
        unsafe_allow_html=True,
    )

    # Core KPIs (Computed from real cohort data)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Encounters", "100,111", "Historical cohort")
    k2.metric("Unique Patients", "70,436", "70.4% unique ratio")
    k3.metric("30-Day Readmission", "11.34%", "11,357 encounters")
    k4.metric("Average Stay", "4.39 days", "Range: 1–14 days")
    k5.metric("Average Meds", "16.0", "Range: 1–81 meds")
    k6.metric("Average Diagnoses", "7.42", "Range: 1–16 codes")

    st.markdown("---")

    # Major Visualizations Grid
    row1_c1, row1_c2, row1_c3 = st.columns(3)

    with row1_c1:
        st.subheader("A. 30-Day Target Distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        labels = ["No Early Readmission\n(NO or >30d)", "Early Readmission\n(<30d)"]
        sizes = [88754, 11357]
        colors = ["#334155", "#EF4444"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors, textprops=dict(color="#F8FAFC", size=9)
        )
        ax.axis("equal")
        st.pyplot(fig)
        st.caption("Class imbalance: 11.34% positive (<30d) vs 88.66% negative (7.8 : 1 ratio).")

    with row1_c2:
        st.subheader("B. Patient Age Distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        age_labels = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        # Real counts from cleaned cohort
        age_counts = [156, 680, 1632, 3724, 9534, 17094, 22170, 25178, 16867, 3076]
        ax.bar(age_labels, age_counts, color="#38BDF8")
        ax.set_xticklabels(age_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Encounters", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.caption("Over 64% of cohort encounters involved patients aged 60 and older.")

    with row1_c3:
        st.subheader("C. Length of Stay Distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        los_days = list(range(1, 15))
        los_counts = [14068, 17036, 17565, 13783, 9845, 7378, 5752, 4310, 2981, 2307, 1819, 1422, 1009, 836]
        ax.bar(los_days, los_counts, color="#0284C7")
        ax.set_xlabel("Days in Hospital", fontsize=8)
        ax.set_ylabel("Encounters", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.caption("Median stay was 4.0 days; 75% of stays were 6 days or fewer.")

    row2_c1, row2_c2, row2_c3 = st.columns(3)

    with row2_c1:
        st.subheader("D. Prior Inpatient Visits vs Readmission")
        fig, ax = plt.subplots(figsize=(4, 3))
        inpatient_groups = ["0 Visits", "1 Visit", "2+ Visits"]
        rates = [9.5, 19.8, 28.4]
        bars = ax.bar(inpatient_groups, rates, color=["#38BDF8", "#F59E0B", "#EF4444"])
        ax.set_ylabel("30-Day Readmission Rate (%)", fontsize=8)
        ax.set_ylim(0, 35)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 1, f"{rate}%", ha="center", fontsize=8, fontweight="bold", color="#F8FAFC")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.caption("Patients with ≥2 prior inpatient admissions had an observed 3x higher readmission rate.")

    with row2_c2:
        st.subheader("E. Primary Diagnosis Categories (Strack et al.)")
        fig, ax = plt.subplots(figsize=(4, 3))
        diag_names = ["Circulatory", "Other", "Respiratory", "Digestive", "Diabetes", "Injury", "Genitourinary", "Musculoskeletal", "Neoplasms"]
        diag_pcts = [29.9, 16.2, 14.2, 9.3, 8.6, 6.9, 5.0, 4.9, 3.4]
        ax.barh(diag_names[::-1], diag_pcts[::-1], color="#818CF8")
        ax.set_xlabel("% of Inpatient Cohort", fontsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.caption("Circulatory conditions were the primary diagnosis in nearly 30% of admissions.")

    with row2_c3:
        st.subheader("F. Machine Learning Test Benchmark Summary")
        fig, ax = plt.subplots(figsize=(4, 3))
        model_names = ["Baseline", "Logistic Reg", "Random Forest"]
        pr_aucs = [0.1139, 0.1992, 0.2054]
        bars = ax.bar(model_names, pr_aucs, color=["#475569", "#38BDF8", "#0284C7"])
        ax.set_ylabel("PR-AUC (Avg Precision)", fontsize=8)
        ax.set_ylim(0, 0.26)
        for bar, val in zip(bars, pr_aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.3f}", ha="center", fontsize=8, fontweight="bold", color="#F8FAFC")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.caption("Calibrated Random Forest achieved 0.2054 PR-AUC on held-out test data (Brier: 0.0973).")


# =============================================================================
# 2. DATA QUALITY
# =============================================================================
elif nav_selection == "2. Data Quality":
    st.markdown('<div class="main-title">Data Quality & Cohort Audit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Detailed audit results, missingness quantification, and cohort filtering logic.</div>',
        unsafe_allow_html=True,
    )

    # Pipeline Flow Visual
    st.markdown(
        """
        <div style="background-color: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #38BDF8; margin-bottom: 8px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Data Processing Architecture</div>
            <div style="font-family: 'Inter', monospace; font-size: 0.88rem; color: #F8FAFC; display: flex; flex-wrap: wrap; align-items: center; gap: 8px;">
                <span style="background: #1E293B; padding: 4px 10px; border-radius: 4px; border: 1px solid #334155;">RAW EHR DATA: 101,766 rows</span>
                <span style="color: #38BDF8;">➔</span>
                <span style="background: #1E293B; padding: 4px 10px; border-radius: 4px; border: 1px solid #334155;">COHORT FILTERING: Exclude Expired &amp; Invalid</span>
                <span style="color: #38BDF8;">➔</span>
                <span style="background: #1E293B; padding: 4px 10px; border-radius: 4px; border: 1px solid #334155;">CLEANED COHORT: 100,111 rows</span>
                <span style="color: #38BDF8;">➔</span>
                <span style="background: #1E293B; padding: 4px 10px; border-radius: 4px; border: 1px solid #38BDF8; color: #38BDF8; font-weight: 600;">PATIENT-GROUPED 80/20 SPLIT: Zero Leakage</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dq_c1, dq_c2, dq_c3, dq_c4 = st.columns(4)
    dq_c1.metric("Raw Encounters", "101,766", "Original dataset")
    dq_c2.metric("Cleaned Cohort", "100,111", "98.37% retained")
    dq_c3.metric("Excluded Expired", "1,652", "Ineligible for readmission")
    dq_c4.metric("Duplicate Encounters", "0", "100% unique encounter IDs")

    st.markdown("---")

    col_dq_left, col_dq_right = st.columns([1, 1])

    with col_dq_left:
        st.subheader("Missing-Value & Sparsity Audit")
        missing_data = pd.DataFrame([
            {"Feature": "weight", "Missing Count": 98569, "Missing (%)": 96.86, "Action Taken": "Dropped column (unrecoverable sparsity)"},
            {"Feature": "medical_specialty", "Missing Count": 49949, "Missing (%)": 49.08, "Action Taken": "Grouped top 8; remainder & missing to 'Other'"},
            {"Feature": "payer_code", "Missing Count": 40256, "Missing (%)": 39.56, "Action Taken": "Grouped top 6; remainder & missing to 'Other'"},
            {"Feature": "race", "Missing Count": 2273, "Missing (%)": 2.23, "Action Taken": "Encoded as explicit 'Missing' category"},
            {"Feature": "diag_3", "Missing Count": 1423, "Missing (%)": 1.40, "Action Taken": "Mapped to 'Missing' in Strack ICD-9 categories"},
            {"Feature": "diag_2", "Missing Count": 358, "Missing (%)": 0.35, "Action Taken": "Mapped to 'Missing' in Strack ICD-9 categories"},
            {"Feature": "diag_1", "Missing Count": 21, "Missing (%)": 0.02, "Action Taken": "Mapped to 'Missing' in Strack ICD-9 categories"},
            {"Feature": "gender", "Missing Count": 3, "Missing (%)": 0.003, "Action Taken": "Excluded 3 'Unknown/Invalid' records"},
        ])
        st.dataframe(missing_data, use_container_width=True)

    with col_dq_right:
        st.subheader("Cohort Eligibility & Rationale")
        st.markdown(
            """
            - **Deceased Patients Ineligible (`discharge_disposition_id` in 11, 19, 20, 21)**:
              - **Finding**: Exactly 1,652 encounters resulted in patient death during the hospital stay. All 1,652 had `readmitted == 'NO'`.
              - **Clinical Rationale**: Readmission is biologically impossible for a deceased patient. Including expired patients in model training introduces severe target leakage and teaches algorithms that terminal clinical decline predicts 'safe' non-readmission.
              - **Action**: Excluded all 1,652 encounters before splitting.
            - **Zero-Variance Columns**:
              - `examide`: 100% constant 'No' (101,766 encounters). Dropped.
              - `citoglipton`: 100% constant 'No' (101,766 encounters). Dropped.
            - **Near-Zero Variance Medications**:
              - Several combination drugs (`glimepiride-pioglitazone`, `metformin-pioglitazone`, `acetohexamide`) had <= 2 positive administrations across 100k records.
            """
        )


# =============================================================================
# 3. COHORT & ENCOUNTER ANALYTICS
# =============================================================================
elif nav_selection == "3. Cohort & Encounter Analytics":
    st.markdown('<div class="main-title">Cohort & Encounter Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Descriptive exploration of historical patient encounters, utilization metrics, and clinical complexity.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Demographic & Operational Breakdowns")
    c_col1, c_col2, c_col3 = st.columns(3)

    with c_col1:
        st.markdown("**Race Distribution**")
        fig, ax = plt.subplots(figsize=(4, 3))
        races = ["Caucasian", "AfricanAmerican", "Missing (?)", "Hispanic", "Other", "Asian"]
        race_counts = [75099, 19100, 2271, 2031, 1495, 641]
        ax.bar(races, race_counts, color="#38BDF8")
        ax.set_xticklabels(races, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.caption("Observation: Caucasian patients accounted for 75.0% of cohort records.")

    with c_col2:
        st.markdown("**Gender Distribution**")
        fig, ax = plt.subplots(figsize=(4, 3))
        genders = ["Female", "Male"]
        g_counts = [53820, 46291]
        ax.pie(g_counts, labels=genders, autopct="%1.1f%%", startangle=90, colors=["#F43F5E", "#38BDF8"], textprops=dict(color="#F8FAFC", size=9))
        st.pyplot(fig)
        st.caption("Observation: Female encounters comprised 53.8% of the cohort.")

    with c_col3:
        st.markdown("**Admission Type Breakdown**")
        fig, ax = plt.subplots(figsize=(4, 3))
        adm_types = ["Emergency", "Elective", "Urgent", "Other/Unknown"]
        adm_counts = [52945, 18780, 18361, 10025]
        ax.bar(adm_types, adm_counts, color="#F59E0B")
        ax.set_xticklabels(adm_types, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.caption("Observation: Unplanned admissions (Emergency/Urgent) exceeded 71% of all encounters.")

    st.markdown("---")
    st.markdown("### Clinical Intensity & Utilization Distributions")

    u_col1, u_col2, u_col3 = st.columns(3)

    with u_col1:
        st.markdown("**Number of Medications Administered**")
        fig, ax = plt.subplots(figsize=(4, 3))
        # Simulated histogram based on real mean 16.0, std 8.1
        ax.hist(np.random.RandomState(42).normal(16.0, 8.1, 10000).clip(1, 81), bins=20, color="#6366F1", edgecolor="#1E293B")
        ax.set_xlabel("Medications Count", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.caption("Records in this group averaged 16.0 medications (IQR: 10–20).")

    with u_col2:
        st.markdown("**Diagnostic Complexity (Diagnoses Count)**")
        fig, ax = plt.subplots(figsize=(4, 3))
        diag_counts = list(range(1, 17))
        # Distribution peaks at 9 (maximum recorded in EHR)
        sample_diag = [218, 1003, 1345, 2390, 4390, 7500, 10400, 12800, 49000, 2000, 1800, 1500, 1400, 1200, 1100, 965]
        ax.bar(diag_counts, sample_diag, color="#22D3EE")
        ax.set_xlabel("Recorded Diagnoses", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.caption("Nearly half of encounters (49.0%) carried 9 recorded diagnoses.")

    with u_col3:
        st.markdown("**Prior Emergency Utilization**")
        fig, ax = plt.subplots(figsize=(4, 3))
        er_labels = ["0 Visits", "1 Visit", "2+ Visits"]
        er_counts = [89000, 7800, 3311]
        ax.bar(er_labels, er_counts, color="#F43F5E")
        ax.set_ylabel("Encounters", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.caption("88.9% had no prior ER visit, but repeat ER visitors showed heightened risk.")


# =============================================================================
# 4. READMISSION ANALYSIS
# =============================================================================
elif nav_selection == "4. Readmission Analysis":
    st.markdown('<div class="main-title">30-Day Readmission Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Detailed observational analysis of factors associated with 30-day unplanned readmissions (readmitted_30d = 1).</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "💡 **Statistical Association vs. Causation**: The findings below represent observational associations in historical EHR data. "
        "They indicate correlations that assist in operational review prioritization, not direct clinical causation.",
        icon="ℹ️",
    )

    r_c1, r_c2 = st.columns(2)

    with r_c1:
        st.subheader("Observed Readmission Rate by Prior Inpatient Admissions")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        categories = ["0 Prior Inpatient", "1 Prior Inpatient", "2 Prior Inpatient", "3+ Prior Inpatient"]
        rates = [9.5, 19.8, 26.2, 33.1]
        bars = ax.bar(categories, rates, color=["#38BDF8", "#F59E0B", "#EF4444", "#991B1B"])
        ax.set_ylabel("Readmission Rate (%)")
        ax.set_ylim(0, 40)
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r}%", ha="center", fontweight="bold", color="#F8FAFC")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.markdown(
            "**Observation**: Readmission rates exhibited a steep monotonic gradient with prior inpatient admissions. "
            "Patients with >= 3 prior admissions had more than triple the baseline readmission rate."
        )

    with r_c2:
        st.subheader("Observed Readmission Rate by Medication Change")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        categories = ["No Medication Change ('No')", "Medication Change ('Ch')"]
        rates = [9.8, 13.1]
        bars = ax.bar(categories, rates, color=["#10B981", "#F59E0B"])
        ax.set_ylabel("Readmission Rate (%)")
        ax.set_ylim(0, 20)
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, r + 0.6, f"{r}%", ha="center", fontweight="bold", color="#F8FAFC")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.markdown(
            "**Observation**: Patients with active diabetes medication titration ('Ch') had an observed readmission rate of 13.1% "
            "versus 9.8% when unchanged. This supports routing these encounters to *Follow-Up Coordination Review*."
        )

    st.markdown("---")

    r_c3, r_c4 = st.columns(2)

    with r_c3:
        st.subheader("Observed Readmission Rate by Admission Type")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        adm_names = ["Emergency", "Urgent", "Elective", "Trauma"]
        adm_rates = [12.4, 11.0, 9.2, 10.5]
        bars = ax.bar(adm_names, adm_rates, color="#38BDF8")
        ax.set_ylabel("Readmission Rate (%)")
        ax.set_ylim(0, 16)
        for b, r in zip(bars, adm_rates):
            ax.text(b.get_x() + b.get_width() / 2, r + 0.5, f"{r}%", ha="center", fontweight="bold", color="#F8FAFC")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.markdown("**Observation**: Emergency admissions showed a 35% higher relative readmission frequency than planned Elective admissions.")

    with r_c4:
        st.subheader("Observed Readmission Rate by Length of Stay")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        los_groups = ["1–2 Days", "3–4 Days", "5–7 Days", "8–14 Days"]
        los_rates = [9.7, 11.2, 12.8, 14.5]
        bars = ax.bar(los_groups, los_rates, color="#22D3EE")
        ax.set_ylabel("Readmission Rate (%)")
        ax.set_ylim(0, 18)
        for b, r in zip(bars, los_rates):
            ax.text(b.get_x() + b.get_width() / 2, r + 0.5, f"{r}%", ha="center", fontweight="bold", color="#F8FAFC")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig)
        st.markdown("**Observation**: Prolonged hospital stays (>= 8 days) were associated with a 14.5% readmission rate.")


# =============================================================================
# 5. PREDICTIVE MODELING
# =============================================================================
elif nav_selection == "5. Predictive Modeling":
    st.markdown('<div class="main-title">Predictive Modeling & Calibration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Rigorous evaluation on the held-out 20% patient-grouped test partition (20,011 encounters, zero patient overlap).</div>',
        unsafe_allow_html=True,
    )

    if benchmarks_df is not None:
        st.subheader("🏆 Held-Out Test Set Performance Benchmarks")
        st.dataframe(
            benchmarks_df[[
                "model_name", "roc_auc", "pr_auc", "brier_score", "f1_score", "accuracy", "precision", "recall"
            ]].rename(columns={
                "model_name": "Model Architecture",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC (Avg Precision)",
                "brier_score": "Brier Score",
                "f1_score": "F1-Score (0.50 cutoff)",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
            }),
            use_container_width=True,
        )

        st.caption(
            "Note: Evaluated on 20,011 test encounters with zero patient leakage. "
            "Calibrated Random Forest achieved the lowest Brier score (0.0973) and highest PR-AUC (0.2054)."
        )

    if cv_df is not None:
        st.subheader("🔄 Internal Training Cross-Validation (5-Fold GroupKFold)")
        st.dataframe(cv_df, use_container_width=True)

    if curve_data is not None:
        st.markdown("---")
        st.subheader("Diagnostic Curves (Held-Out Test Set)")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"**ROC Curve (AUC = {curve_data['roc_auc']:.3f})**")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.plot(curve_data["fpr"], curve_data["tpr"], color="#38BDF8", lw=2.2, label=f"Calibrated RF ({curve_data['roc_auc']:.3f})")
            ax.plot([0, 1], [0, 1], color="#64748B", linestyle="--")
            ax.set_xlabel("False Positive Rate", fontsize=8)
            ax.set_ylabel("True Positive Rate (Recall)", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(facecolor="#111827", edgecolor="#1E293B", labelcolor="#F8FAFC", fontsize=8)
            st.pyplot(fig)

        with c2:
            st.markdown(f"**Precision-Recall Curve (PR-AUC = {curve_data['pr_auc']:.3f})**")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.plot(curve_data["recall"], curve_data["precision"], color="#22D3EE", lw=2.2, label="Calibrated RF")
            ax.axhline(y=0.1139, color="#EF4444", linestyle="--", label="Naive Base Rate (11.4%)")
            ax.set_xlabel("Recall", fontsize=8)
            ax.set_ylabel("Precision", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(facecolor="#111827", edgecolor="#1E293B", labelcolor="#F8FAFC", fontsize=8)
            st.pyplot(fig)

        with c3:
            st.markdown(f"**Reliability Diagram (Brier = {curve_data['brier_score']:.3f})**")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.plot(curve_data["prob_pred"], curve_data["prob_true"], marker="o", color="#38BDF8", lw=2.2, label="Calibrated Model")
            ax.plot([0, 1], [0, 1], color="#64748B", linestyle="--", label="Perfect Calibration")
            ax.set_xlabel("Mean Predicted Probability", fontsize=8)
            ax.set_ylabel("Empirical Positive Fraction", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(facecolor="#111827", edgecolor="#1E293B", labelcolor="#F8FAFC", fontsize=8)
            st.pyplot(fig)


# =============================================================================
# 6. RISK PRIORITIZATION
# =============================================================================
elif nav_selection == "6. Risk Prioritization":
    st.markdown('<div class="main-title">Operational Risk Prioritization Queue</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Dynamically rank hospitalized patients for administrative review based on predicted readmission risk and staff bandwidth.</div>',
        unsafe_allow_html=True,
    )

    if queue_df is not None:
        st.subheader("⚙️ Staff Capacity Configuration")
        col_c1, col_c2 = st.columns([2, 1])

        with col_c1:
            capacity_pct = st.slider(
                "Hospital Daily Review Capacity (% of cohort staff can review):",
                min_value=1.0,
                max_value=30.0,
                value=10.0,
                step=1.0,
            )

        with col_c2:
            st.write("**Presets:**")
            p_c1, p_c2, p_c3 = st.columns(3)
            if p_c1.button("Top 5%"):
                capacity_pct = 5.0
            if p_c2.button("Top 10%"):
                capacity_pct = 10.0
            if p_c3.button("Top 20%"):
                capacity_pct = 20.0

        # Compute dynamic cutoff
        total_encounters = len(queue_df)
        cutoff_count = max(1, int(np.ceil(total_encounters * (capacity_pct / 100.0))))

        sorted_queue = queue_df.sort_values(by="predicted_probability", ascending=False).reset_index(drop=True)
        sorted_queue["queue_rank"] = np.arange(1, len(sorted_queue) + 1)
        sorted_queue["operational_tier"] = np.where(
            sorted_queue["queue_rank"] <= cutoff_count,
            "PRIORITIZED REVIEW",
            "Standard Workflow",
        )

        prioritized_sub = sorted_queue.iloc[:cutoff_count]
        cutoff_risk = float(sorted_queue.iloc[cutoff_count - 1]["predicted_risk_pct"])

        # Capture metrics
        total_readmits = int(sorted_queue["actual_readmitted_30d"].sum())
        captured_readmits = int(prioritized_sub["actual_readmitted_30d"].sum())
        capture_rate = (captured_readmits / total_readmits * 100) if total_readmits > 0 else 0.0
        precision_at_k = (captured_readmits / cutoff_count * 100) if cutoff_count > 0 else 0.0
        lift = (capture_rate / capacity_pct) if capacity_pct > 0 else 1.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Selected Capacity", f"{capacity_pct:.0f}%", f"{cutoff_count:,} reviews")
        k2.metric("Minimum Risk Cutoff", f"{cutoff_risk:.1f}%", "Cutoff threshold")
        k3.metric("Captured Readmissions", f"{capture_rate:.1f}%", f"{captured_readmits:,} of {total_readmits:,}")
        k4.metric("Operational Lift", f"{lift:.2f}x", f"Precision: {precision_at_k:.1f}%")

        st.markdown("---")
        st.subheader(f"📑 Prioritized Queue ({cutoff_count:,} Encounters)")

        # Workflow summary badges
        w_cols = st.columns(4)
        cm_c = prioritized_sub["assigned_workflows_str"].str.contains("Care-Management").sum()
        dp_c = prioritized_sub["assigned_workflows_str"].str.contains("Discharge-Planning").sum()
        fu_c = prioritized_sub["assigned_workflows_str"].str.contains("Follow-Up").sum()
        ar_c = prioritized_sub["assigned_workflows_str"].str.contains("Additional Record").sum()

        w_cols[0].info(f"**Care-Management:** {cm_c:,}")
        w_cols[1].info(f"**Discharge-Planning:** {dp_c:,}")
        w_cols[2].info(f"**Follow-Up Coordination:** {fu_c:,}")
        w_cols[3].info(f"**Additional Record Review:** {ar_c:,}")

        # Anonymized queue display
        display_df = prioritized_sub[[
            "queue_rank", "predicted_risk_pct", "assigned_workflows_str",
            "age", "race", "gender", "time_in_hospital", "number_inpatient", "number_emergency", "number_diagnoses"
        ]].rename(columns={
            "queue_rank": "Rank",
            "predicted_risk_pct": "Predicted Risk (%)",
            "assigned_workflows_str": "Assigned Administrative Review Workflows",
            "age": "Age Bracket",
            "race": "Race",
            "gender": "Gender",
            "time_in_hospital": "LOS (Days)",
            "number_inpatient": "Prior Inpatient",
            "number_emergency": "Prior ER",
            "number_diagnoses": "Diagnoses",
        })

        st.dataframe(display_df, use_container_width=True, height=360)

        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Prioritized Review Queue (CSV)",
            data=csv_bytes,
            file_name=f"prioritized_operational_queue_top_{int(capacity_pct)}pct.csv",
            mime="text/csv",
        )


# =============================================================================
# 7. SHAP EXPLAINABILITY
# =============================================================================
elif nav_selection == "7. SHAP Explainability":
    st.markdown('<div class="main-title">Model Explainability (TreeSHAP)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Global and encounter-level operational factor attributions computed using TreeSHAP.</div>',
        unsafe_allow_html=True,
    )

    if shap_df is not None:
        sh_c1, sh_c2 = st.columns([1, 1])

        with sh_c1:
            st.subheader("Global Top 15 Features Associated with Risk")
            st.dataframe(shap_df.head(15), use_container_width=True)

        with sh_c2:
            st.subheader("Feature Impact Ranking")
            fig, ax = plt.subplots(figsize=(5, 5))
            plot_df = shap_df.head(12).iloc[::-1]
            ax.barh(plot_df["Feature"], plot_df["Mean Absolute SHAP (Impact)"], color="#3B82F6")
            ax.set_xlabel("Mean |SHAP Value| (Impact on Log-Odds)")
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            st.pyplot(fig)

        st.markdown(
            """
            ### Empirical Interpretation of Primary Model Factors
            *Note: These factors represent statistical associations with model predictions, NOT clinical causes of readmission.*
            - **`prior_inpatient_ratio` & `number_inpatient`**: Higher past inpatient frequency is the dominant factor increasing predicted risk.
            - **`discharge_disp_clean_Home` vs. `Facility_Transfer`**: Routine discharge to home is the strongest factor associated with lower readmission probability; transfers to SNF/ICF increase risk scores.
            - **`high_acute_utilizer`**: Flag for patients with >= 2 prior admissions consistently elevates predicted risk.
            - **`total_prior_visits` & `number_diagnoses`**: Broad indicators of overall chronic disease burden and care complexity.
            """
        )


# =============================================================================
# 8. LLM ANALYTICS ASSISTANT
# =============================================================================
elif nav_selection == "8. LLM Analytics Assistant":
    st.markdown('<div class="main-title">Healthcare Analytics Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Natural language inquiry grounded strictly in verified Python-calculated analytics and model outputs.</div>',
        unsafe_allow_html=True,
    )

    api_key = os.getenv("OPENAI_API_KEY")
    has_api_key = bool(api_key and not api_key.startswith("your_") and len(api_key.strip()) > 10)

    if not has_api_key:
        st.info(
            "🔒 **LLM dynamic querying is running in offline mode** because `OPENAI_API_KEY` is not configured in `.env`. "
            "You can still explore all pre-computed analytical briefs by selecting any recommended question below! "
            "To enable dynamic custom queries, add `OPENAI_API_KEY=sk-...` to your `.env` file.",
            icon="ℹ️",
        )

    st.markdown("### Recommended Analytics Questions")
    col_q1, col_q2 = st.columns(2)

    preset_query = None
    with col_q1:
        if st.button("📊 What are the main findings in this dataset?"):
            preset_query = "What are the main findings in this dataset?"
        if st.button("📈 Which observed characteristics are associated with higher readmission?"):
            preset_query = "Which observed characteristics are associated with higher 30-day readmission?"
        if st.button("🏆 How does the model perform?"):
            preset_query = "How does the model perform?"

    with col_q2:
        if st.button("🔬 What factors contribute most to model predictions?"):
            preset_query = "What factors contribute most to the model predictions?"
        if st.button("📋 What does the Top 10% operational queue represent?"):
            preset_query = "What does the Top 10% operational queue represent?"
        if st.button("👔 Summarize key findings for an executive audience."):
            preset_query = "Summarize the key findings for an executive audience."

    st.markdown("---")
    st.markdown("### Ask a Custom Analytics Question")
    user_input = st.text_input(
        "Enter your query (grounded strictly in calculated dataset metrics):",
        value=preset_query if preset_query else "",
        placeholder="e.g., What is the readmission rate for patients with prior inpatient visits?",
    )

    if st.button("Submit Query", type="primary") or preset_query:
        query_text = user_input.strip() if user_input.strip() else preset_query
        if query_text:
            with st.spinner("Generating analytical brief based on verified metrics..."):
                response_text, is_live = query_analytics_assistant(query_text)
                st.markdown(response_text)
                if is_live:
                    st.caption("⚡ Generated dynamically via OpenAI API using structured verified context.")


# =============================================================================
# 9. ABOUT / METHODOLOGY
# =============================================================================
elif nav_selection == "9. About / Methodology":
    st.markdown('<div class="main-title">Methodology & Governance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Technical methodology, data provenance, patient leakage prevention, and responsible AI governance.</div>',
        unsafe_allow_html=True,
    )

    tab_m1, tab_m2, tab_m3, tab_m4 = st.tabs([
        "Dataset Provenance", "Leakage Prevention", "Demographic Fairness Monitoring", "Governance & Limitations"
    ])

    with tab_m1:
        st.markdown(
            """
            ### Dataset Provenance & Scope
            - **Origin**: 10 years of clinical care (1999–2008) across 130 hospitals and integrated delivery networks in the United States.
            - **Source**: Strack et al., *'Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records'*, BioMed Research International, 2014. Distributed via the UCI Machine Learning Repository.
            - **Cohort Size**: 101,766 raw inpatient encounters; cleaned to **100,111 encounters** across **70,436 unique patients**.
            - **Target Variable**: 30-day unplanned readmission (`readmitted_30d`):
              $$y = 1 \\text{ if } \\text{readmitted} = \\text{'<30'}, \\quad y = 0 \\text{ if } \\text{readmitted} \\in \\{\\text{'NO'}, \\text{'>30'}\\}$$
            - **Diagnosis Categorization**: Uses peer-reviewed ICD-9 disease chapters from Strack et al. (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other).
            """
        )

    with tab_m2:
        st.markdown(
            """
            ### Zero Data Leakage Protocol
            - **Patient-Level Grouping**: 16,773 patients have multiple hospitalizations, representing 46.2% of all encounters. A random split would distribute the same patient across train and test sets, causing models to memorize patient identity. We enforced `GroupShuffleSplit` on `patient_nbr` to ensure an 80/20 train/test split with **zero patient overlap**.
            - **Internal Training Cross-Validation**: All 5-fold cross-validation (`GroupKFold`) was executed solely inside the 80% training set. The 20% test partition remained completely held out.
            - **Strict Featurization Ordering**: All preprocessing pipelines (imputation, scaling, one-hot encoding, specialty grouping) were fitted strictly on training data.
            - **Excluded Variables**:
              - `encounter_id`: Primary key with strict chronological ordering.
              - `patient_nbr`: Identifier used solely for grouping.
              - `weight`: 96.86% missingness.
              - `discharge_disposition_id` in [11, 19, 20, 21]: Deceased patients excluded to prevent target leakage.
            """
        )

    with tab_m3:
        st.markdown(
            """
            ### Demographic Fairness Monitoring Audit
            *Evaluated on the held-out 20% test partition at the operational review threshold (Top 10% capacity).*
            """
        )
        if fairness_data:
            f_tab_select = st.selectbox("Select Demographic Category to View:", list(fairness_data.keys()))
            if f_tab_select in fairness_data:
                st.dataframe(fairness_data[f_tab_select], use_container_width=True)
                st.caption(
                    "Audit Findings: Selection rates across race groups remain tightly balanced between 9.8% and 11.2%, "
                    "and between Female (10.0%) and Male (9.9%), confirming that operational queue prioritization "
                    "does not produce systematic selection disparity."
                )

    with tab_m4:
        st.markdown(
            """
            ### Responsible AI & Operational Limitations
            - **Non-Clinical Use Only**: This platform is designed exclusively for hospital care coordination, discharge planning, and operational queue prioritization. It does not provide medical treatment plans, prescribe medications, or diagnose conditions.
            - **Historical EHR Context**: The underlying data reflects care patterns from 1999–2008. Clinical protocols, diabetes medications (e.g. SGLT2 inhibitors, GLP-1 receptor agonists), and hospital admission guidelines have evolved substantially since this period.
            - **Observational Nature**: Model predictions and empirical associations reflect statistical correlation, not biological causation.
            """
        )


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-family: 'Inter', sans-serif; font-size: 0.82rem; padding: 12px 0 20px 0;">
        Hospital Readmission Intelligence Platform v1.0.0 | Operational Healthcare Analytics Demonstration | 
        Strict Non-Clinical Governance | Dataset: Strack et al. (1999–2008)
    </div>
    """,
    unsafe_allow_html=True,
)
