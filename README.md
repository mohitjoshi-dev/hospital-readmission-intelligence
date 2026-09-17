# Hospital Readmission Intelligence

### Operational Healthcare Analytics & Readmission-Risk Decision-Support Platform

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-8%20passed-brightgreen.svg)](tests/)
[![Dashboard: Streamlit](https://img.shields.io/badge/dashboard-streamlit-FF4B4B.svg)](dashboard/app.py)

---

## ⚠️ Platform Governance & Responsible AI Disclaimer

> **CRITICAL NOTICE**:
> This platform is an **operational healthcare analytics and decision-support demonstration** utilizing historical electronic health record (EHR) data (1999–2008) across 130 US hospitals (Strack et al., 2014).
> 
> **It is NOT a clinical diagnostic system, medical treatment system, prescribing platform, or clinical decision-making system.**
> 
> Its sole purpose is to assist hospital operations and care coordination teams (discharge planners, care managers, follow-up coordinators) in **prioritizing administrative chart reviews and transitional care workflows** under constrained daily staff bandwidth. 
> 
> The platform does **not** generate medical diagnoses, recommend patient-specific clinical interventions, or prescribe medications. Automated text summaries and the LLM Analytics Assistant are strictly confined to explaining aggregated statistical outputs and deterministic model metrics supplied directly by Python.

---

## 1. Executive Summary & Business Problem

Under the Centers for Medicare & Medicaid Services (CMS) **Hospital Readmissions Reduction Program (HRRP)**, hospitals face substantial financial penalties for excess 30-day unplanned readmissions. For complex chronic conditions such as diabetes, patient discharge coordination requires extensive administrative follow-up. 

However, hospital care management and discharge planning staff face **finite operational capacity**—typically reviewing only 5% to 20% of daily discharged patients. Without risk stratification, staff review patients essentially at random or based on intuitive heuristics.

**Hospital Readmission Intelligence** bridges this gap:
- **Calibrated Predictive Risk Scoring**: Estimates true continuous 30-day readmission risk using leak-free, patient-grouped machine learning models.
- **Dynamic Capacity Triage**: Enables hospital teams to set their available daily review bandwidth (e.g., Top 5%, Top 10%, Top 20%), generating a prioritized queue that delivers a **2.12x operational lift** over random selection.
- **Administrative Workflow Routing**: Automatically assigns prioritized discharges to four non-clinical operational workflows:
  - *Care-Management Review* (past acute hospital utilization)
  - *Discharge-Planning Review* (complex non-home disposition)
  - *Follow-Up Coordination Review* (active diabetic medication adjustments)
  - *Additional Record Review* (high diagnostic complexity)
- **Transparent Factor Attribution**: Explains model risk scores using TreeSHAP to reveal key operational drivers without black-box opacity.
- **Demographic Fairness Monitoring**: Continuously audits operational review selection rates across Race, Gender, and Age to ensure review parity.
- **LLM Analytics Assistant**: Translates verified dataset metrics into executive briefs with zero statistical hallucination.

---

## 2. Dataset Provenance & Cohort Characteristics

- **Origin**: 10 years of clinical care (1999–2008) across 130 hospitals and integrated healthcare delivery networks in the United States.
- **Publication**: Strack et al., *"Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records"*, BioMed Research International, 2014 (distributed via the UCI Machine Learning Repository).
- **Original Dataset**: 101,766 inpatient encounters across 71,518 unique patients, with 50 recorded demographic, clinical utilization, medication, and billing variables.
- **Final Cleaned Cohort**: **100,111 encounters** across **70,436 unique patients** (98.37% retention rate).
- **Target Event Rate**: **11.34%** (11,357 unplanned 30-day readmissions).

---

## 3. The 4-Tier Healthcare Analytics Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     4-TIER ANALYTICS FRAMEWORK                          │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
     ┌─────────────────────────────┼─────────────────────────────┐
     ▼                             ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│   TIER 1: DESCRIPTIVE   │   │   TIER 2: DIAGNOSTIC    │   │   TIER 3: PREDICTIVE    │
├─────────────────────────┤   ├─────────────────────────┤   ├─────────────────────────┤
│ • Cohort profiling      │   │ • Statistical testing   │   │ • Patient-Grouped 80/20 │
│ • Utilization trends    │   │ • Prior inpatient rates │   │ • Logistic Regression   │
│ • HbA1c testing gaps    │   │ • Med change impact     │   │ • Calibrated Random     │
│ • CCS diagnosis maps    │   │ • Global TreeSHAP rank  │   │   Forest (Brier: 0.097) │
└─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
                                                                 │
                                                                 ▼
                                                    ┌─────────────────────────┐
                                                    │   TIER 4: PRESCRIPTIVE  │
                                                    ├─────────────────────────┤
                                                    │ • Capacity queues       │
                                                    │   (Top 5%, 10%, 20%)    │
                                                    │ • Administrative review │
                                                    │   workflow routing      │
                                                    │ • LLM Analytics briefs  │
                                                    └─────────────────────────┘
```

1. **Tier 1: Descriptive Analytics**: Population demographics, length of stay, prior utilization patterns, diagnostic complexity, and glycemic testing frequency (quantified that 83.2% of inpatient diabetic records lacked HbA1c testing).
2. **Tier 2: Diagnostic Analytics**: Statistical associations and empirical risk gradients (prior inpatient admissions gradient: 9.5% for 0 vs 28.4% for $\ge 2$; medication adjustments: 13.1% vs 9.8%). Clearly distinguished correlation from clinical causation.
3. **Tier 3: Predictive Analytics**: Supervised classification benchmarked against a held-out patient-grouped test set (Majority Baseline, Logistic Regression, Random Forest, Calibrated Random Forest) with PR-AUC, ROC-AUC, and Brier calibration curves.
4. **Tier 4: Prescriptive Analytics**: Operational prioritization engine matching hospital daily review capacity to predicted risk, assigning operational review tags, and providing local SHAP factor attribution.

---

## 4. Rigorous Data Cleaning & Leakage Prevention Protocol

### A. Clinical Cohort Ineligibility Exclusions
- **Deceased Patients Excluded**: Exactly **1,652 encounters** had `discharge_disposition_id` in `[11, 19, 20, 21]` (Expired).
- **Target Leakage Prevention**: In the raw data, 100% of expired encounters had `readmitted == 'NO'`. Including deceased patients introduces fatal target leakage—training models to associate terminal organ failure with "safe" non-readmission. All 1,652 encounters were excluded prior to modeling.
- **Demographic Anomalies**: Excluded 3 records with `gender == 'Unknown/Invalid'`.

### B. Zero Patient Leakage (Patient-Grouped 80/20 Split)
- **The Threat**: 16,773 patients in the dataset had multiple hospitalizations (accounting for 46.2% of encounters). Standard random splitting leaks patient identity, genetic traits, and unobserved chronic baselines across train and test sets.
- **The Mitigation**: Partitioned using `GroupShuffleSplit` on `patient_nbr`:
  - **Training Cohort**: 80,100 encounters (56,348 unique patients, 11.33% positive rate).
  - **Held-Out Test Cohort**: 20,011 encounters (14,088 unique patients, 11.39% positive rate).
  - **Zero Patient Overlap**: $\text{Train Patients} \cap \text{Test Patients} = \emptyset$.
- **Train-Only Cross-Validation**: All 5-fold cross-validation (`GroupKFold` on `patient_nbr`) was executed strictly **inside** the 80% training set.
- **Featurization Ordering**: All transformers (StandardScaler, OneHotEncoder, specialty frequency grouping) were fitted strictly on training data.

### C. Standardized Diagnosis Groupings (Strack et al., 2014)
ICD-9 diagnosis codes (`diag_1`, `diag_2`, `diag_3`) are mapped directly into the peer-reviewed clinical intervals published in the original study (*Strack et al., 2014, Table 2*):
- **Circulatory**: 390–459, 785 (29.9% of cohort)
- **Respiratory**: 460–519, 786 (14.2% of cohort)
- **Digestive**: 520–579, 787 (9.3% of cohort)
- **Diabetes**: 250.xx (8.6% of cohort)
- **Injury & Poisoning**: 800–999 (6.9% of cohort)
- **Musculoskeletal**: 710–739 (4.9% of cohort)
- **Genitourinary**: 580–629, 788 (5.0% of cohort)
- **Neoplasms**: 140–239 (3.4% of cohort)
- **Other / Supplementary**: V and E codes, and all remaining codes (16.2% of cohort)

---

## 5. Model Evaluation & Benchmark Results

### A. Internal Training Cross-Validation (5-Fold GroupKFold)
| Model Architecture | CV ROC-AUC (Mean ± Std) | CV PR-AUC (Mean ± Std) | CV Brier Score (Mean) |
| :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | 0.5000 ± 0.0000 | 0.1133 ± 0.0026 | 0.1133 |
| **Logistic Regression Baseline** | 0.6586 ± 0.0058 | 0.2025 ± 0.0052 | 0.2285 |
| **Random Forest Classifier** | **0.6667 ± 0.0035** | **0.2115 ± 0.0058** | **0.2019** |

### B. Held-Out 20% Test Set Benchmark (20,011 encounters, 14,088 patients)
| Model Architecture | ROC-AUC | PR-AUC (Avg Precision) | Brier Calibration Score | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | 0.5000 | 0.1139 | 0.1139 | 0.0000 |
| **Logistic Regression Baseline** | 0.6453 | 0.1992 | 0.2284 | 0.2648 |
| **Random Forest Classifier** | 0.6576 | 0.2030 | 0.2047 | 0.2777 |
| **Calibrated Random Forest** | **0.6572** | **0.2054** | **0.0973** (Best) | 0.0000\* |

*\*Note on Calibration: Applying sigmoid calibration via `CalibratedClassifierCV` optimized the Brier score to 0.0973, aligning continuous model probabilities with true empirical event rates. Discrete F1 at 0.50 cutoff is 0 because empirical probabilities rarely exceed 50% given the 11.34% base event rate; operational triage therefore operates via capacity-based queue cutoffs.*

---

## 6. Operational Capacity Prioritization & Lift

Instead of arbitrary, rigid risk thresholds, the platform implements **configurable operational capacity queues** matching hospital staff bandwidth:

| Capacity Preset | Cutoff Risk Probability | Prioritized Reviews | Captured Readmissions | Precision @ Capacity | Operational Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 5% Capacity** | $\ge 22.97\%$ | 1,001 encounters | 293 patients | **29.27%** | **2.57x** |
| **Top 10% Capacity** | $\ge 19.36\%$ | 2,002 encounters | 484 patients | **24.18%** | **2.12x** |
| **Top 20% Capacity** | $\ge 15.21\%$ | 4,003 encounters | 847 patients | **21.16%** | **1.86x** |

### Non-Clinical Operational Review Routing
Prioritized patients are automatically tagged for specific administrative review workflows:
1. **Care-Management Review**: Flagged for acute prior utilization ($\ge 2$ prior inpatient admissions or $\ge 2$ emergency visits).
2. **Discharge-Planning Review**: Flagged for non-routine discharge dispositions (transfers to SNF, ICF, or Home Health).
3. **Follow-Up Coordination Review**: Flagged for active diabetes medication titration (`change == 'Ch'` or insulin dose changes).
4. **Additional Record Review**: Flagged for high diagnostic burden ($\ge 9$ recorded diagnoses or multiple comorbidity clusters).

---

## 7. Model Explainability & Factor Attribution (TreeSHAP)

Global feature importance was computed across test encounters using `shap.TreeExplainer`:
1. `prior_inpatient_ratio` (Mean |SHAP|: 0.0253) — Proportion of prior hospital visits that were inpatient admissions.
2. `discharge_disp_clean_Home` (Mean |SHAP|: 0.0218) — Routine home discharge (associated with lower risk).
3. `number_inpatient` (Mean |SHAP|: 0.0205) — Inpatient stays in preceding 12 months.
4. `high_acute_utilizer` (Mean |SHAP|: 0.0169) — Flag for $\ge 2$ prior acute admissions.
5. `discharge_disp_clean_Facility_Transfer` (Mean |SHAP|: 0.0163) — Transfer to SNF/ICF facility.
6. `total_prior_visits` & `number_diagnoses` — Chronic comorbidity complexity indicators.

---

## 8. Demographic Fairness Monitoring

Fairness was audited across Race, Gender, and Age at the operational review cutoff (Top 10% capacity):
- **Race Parity**: Selection rates remain closely balanced across groups: Caucasian (**9.85%**), African American (**11.20%**), Hispanic (**12.53%**), Asian (**6.50%**).
- **Gender Parity**: Selection rates are nearly identical: Female (**10.04%**) vs. Male (**9.94%**).
- **Age Parity**: Selection rates across age brackets: $>70$ years (**9.64%**), $50-70$ years (**9.32%**), $<50$ years (**12.56%**).

*Note: Fairness tables serve strictly as administrative monitoring tools; demographic features are never used as hard decision rules.*

---

## 9. LLM Analytics Assistant

The platform integrates an **LLM Analytics Assistant** that translates complex dataset statistics and model outputs into natural language executive briefs:
- **Zero Hallucination Guardrail**: The LLM does **not** calculate or infer numbers directly; Python compiles verified, structured context from real analytical artifacts and passes it as the sole factual source of truth.
- **Non-Clinical Guardrail**: Strict prompt constraints prevent the model from issuing clinical diagnoses, drug prescriptions, or medical treatment plans.
- **Graceful Degradation**: If `OPENAI_API_KEY` is not present, the assistant runs in offline mode without crashing, providing instant deterministic briefs for all recommended queries.
- **Privacy First**: Zero raw patient identifiers, records, or PII are ever sent to external APIs.

---

## 10. Complete 9-Page Streamlit Experience

The interactive dashboard (`dashboard/app.py`) is organized into a complete healthcare data intelligence experience:
1. **Executive Overview**: High-level KPI cards, real cohort charts, and model benchmark summary.
2. **Data Quality**: Audit numbers, missingness tables, mortality exclusion rationale, and pipeline flowchart.
3. **Cohort & Encounter Analytics**: Demographic breakdowns, utilization distributions, and empirical observations.
4. **Readmission Analysis**: 30-day readmission deep dive across utilization, medication changes, and stay length.
5. **Predictive Modeling**: Held-out test metrics, ROC curve, PR curve, reliability diagram, and cross-validation table.
6. **Risk Prioritization**: Interactive capacity slider (1%–30%), preset buttons, live KPI cards, and CSV export.
7. **SHAP Explainability**: Global feature importance chart, metric table, and factor interpretations.
8. **LLM Analytics Assistant**: Interactive Q&A interface with 6 suggested questions, custom query box, and OpenAI integration.
9. **About / Methodology**: Full documentation of data provenance, zero-leakage protocol, fairness monitoring, and governance.

---

## 11. Installation & Execution Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows PowerShell / Linux / macOS

### 1. Environment Setup
```powershell
# Create and activate virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install lean dependencies
.\.venv\Scripts\pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional for LLM)
```powershell
# Copy the example environment configuration
Copy-Item .env.example .env

# Open .env and add your OpenAI API key if desired:
# OPENAI_API_KEY=sk-...
```
*(Note: All data, analytics, modeling, queue triage, and offline assistant features work 100% without an API key!)*

### 3. Run Automated Tests
```powershell
.\.venv\Scripts\pytest tests/ -v
```
All 8 automated tests will execute and validate schema integrity, mortality exclusion, zero patient leakage, Strack ICD-9 mappings, queue capacity cutoffs, workflow routing, context compilation, and LLM offline fallback.

### 4. Run Training & Evaluation Pipeline
```powershell
.\.venv\Scripts\python src/train_and_evaluate.py
```

### 5. Launch Interactive Streamlit Dashboard (Local)
```powershell
.\.venv\Scripts\streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to explore the platform.

### 6. Streamlit Deployment Guide (Cloud / Production)

#### A. Streamlit Community Cloud Deployment
1. Push this repository to GitHub (ensure raw CSVs and `.env` are excluded via `.gitignore`).
2. Log in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **New App** and select your repository:
   - **Repository**: `YourUsername/Hospital-Readmission-Intelligence`
   - **Branch**: `main` (or `master`)
   - **Main file path**: `dashboard/app.py`
4. In **Advanced Settings** > **Secrets**, optionally provide your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
5. Click **Deploy!** The pre-generated reports and trained models will load instantly.

#### B. Production Server / VM Deployment (Linux/Docker)
```bash
# Run headlessly in production behind Nginx/reverse proxy
streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection true
```

---

## 12. Automated Test Suite Summary

The automated test suite (`pytest tests/ -v`) provides continuous verification across all core platform modules:
- `test_ids_mapping_structure`: Verifies multi-section relational lookup parsing.
- `test_mortality_exclusion_and_gender_clean`: Verifies expired patients and demographic anomalies are cleanly filtered.
- `test_patient_grouped_split_zero_leakage`: Asserts zero patient overlap between train and test partitions.
- `test_strack_icd9_groupings`: Verifies exact ICD-9 decimal range mapping into published Strack categories.
- `test_operational_queue_capacity_cutoff`: Verifies dynamic capacity cutoff precision and queue ranking.
- `test_operational_workflow_assignment`: Verifies administrative review tags route correctly.
- `test_compile_verified_analytical_context`: Asserts that context compiles real calculated metrics without PII.
- `test_offline_analytics_assistant_graceful_handling`: Verifies assistant functions cleanly without an API key.

---

## 13. Limitations & Future Directions

- **Historical Timeframe**: The dataset spans 1999–2008. Contemporary inpatient diabetes care increasingly utilizes continuous glucose monitors (CGMs), GLP-1 receptor agonists, and SGLT2 inhibitors, which are not represented in this historical cohort.
- **Socioeconomic Determinants of Health (SDoH)**: The dataset lacks granular social determinant variables (transportation access, food security, health literacy) that strongly influence post-discharge follow-up adherence.
- **Single Health System Adaptation**: While trained across 130 hospitals, local clinical workflow rules and EHR configurations should be calibrated per health system before production deployment.
