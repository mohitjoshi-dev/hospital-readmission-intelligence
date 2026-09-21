# 🏥 Hospital Readmission Intelligence

An operational healthcare analytics and decision-support platform designed to predict 30-day unplanned readmission risk and prioritize hospital care-coordination workflows under constrained staff capacity. Built on historical Electronic Health Record (EHR) data covering 10 years of clinical care across 130 U.S. hospitals, the system combines leak-free machine learning, calibrated probability estimation, TreeSHAP explainability, and an interactive 9-page Streamlit intelligence dashboard.

> **Operational Decision-Support Disclaimer**: This platform is an operational analytics and workflow prioritization demonstration using historical EHR data. It is **NOT** a clinical diagnostic, treatment, medication prescription, or medical decision-making system. All recommendations are strictly limited to hospital administrative review workflows (care management, discharge planning, follow-up coordination, and additional record review).

---

## 🚀 Dashboard

- **Local Dashboard**: [http://localhost:8501](http://localhost:8501)
- **GitHub Repository**: [https://github.com/mohitjoshi-dev/hospital-readmission-intelligence](https://github.com/mohitjoshi-dev/hospital-readmission-intelligence)

---

## 📊 Dataset

- **Dataset**: Diabetes 130-US Hospitals for Years 1999–2008
- **Original Cohort**: 101,766 inpatient encounters across 71,518 unique patients (50 attributes)
- **Final Cleaned Cohort**: 100,111 eligible inpatient encounters across 70,436 unique patients
- **Observed 30-Day Readmission Rate**: 11.34% (11,357 encounters)
- **Scope**: 130 U.S. hospitals and integrated delivery networks (1999–2008 period)
- **Official Source**: [UCI Machine Learning Repository — Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)

*Note: Raw healthcare CSV files (`diabetic_data.csv`, `IDS_mapping.csv`) are intentionally excluded from GitHub in accordance with healthcare data governance and repository storage best practices. Download the raw dataset from the official UCI link above and place files in `data/raw/` or `raw/`.*

---

## 🎯 Objectives

- **Data Quality & Cohort Analysis**: Quantify missingness, drop unrecoverable and zero-variance features, and filter clinically ineligible records.
- **Readmission Pattern Analysis**: Uncover empirical clinical and utilization patterns associated with 30-day readmissions.
- **Leakage-Controlled ML Prediction**: Enforce strict patient-grouped partitioning to eliminate cross-set identity memorization.
- **Model Evaluation & Calibration**: Benchmark candidate algorithms and calibrate probabilities to produce dependable continuous risk estimates.
- **SHAP Explainability**: Provide transparent global and encounter-level feature attributions to explain risk drivers.
- **Operational Risk Prioritization**: Deliver dynamic, capacity-based triage queues that maximize readmission capture within daily staff limits.
- **LLM-Based Analytics Assistance**: Support natural language analytical inquiry grounded strictly in verified Python calculations.

---

## 🔍 Analytics & Methodology

- **Data Cleaning & Cohort Filtering**: Replaced missing tokens (`'?'` $\to$ `NaN`), dropped unrecoverable columns (`weight` at 96.86% missingness, zero-variance `examide` and `citoglipton`), bucketed high-cardinality features (`payer_code`, `medical_specialty`), and excluded 1,652 deceased encounters (`discharge_disposition_id` in 11, 19, 20, 21). Because readmission is biologically impossible after death (100% had `readmitted == 'NO'`), retaining expired encounters introduces fatal target leakage.
- **Target Definition**: Binary 30-day unplanned readmission target:
  $$y = 1 \text{ if } \text{readmitted} = \text{'<30'}, \quad y = 0 \text{ if } \text{readmitted} \in \{\text{'NO'}, \text{'>30'}\}$$
- **Patient-Grouped 80/20 Partitioning**: Because 16,773 patients have multiple hospitalizations (46.2% of all admissions), random splitting causes data leakage. Using `GroupShuffleSplit` on `patient_nbr`, the cohort is partitioned into 80,100 training encounters (56,348 patients) and 20,011 testing encounters (14,088 patients) with **zero patient overlap**.
- **Clinical Feature Engineering**: Mapped primary, secondary, and tertiary diagnoses into peer-reviewed ICD-9 chapters from Strack et al. (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other) and engineered chronic utilization velocity indicators (`prior_inpatient_ratio`, `high_acute_utilizer`, `total_prior_visits`).
- **Descriptive & Diagnostic Analytics**: Identified steep risk gradients across prior inpatient admissions (9.5% for 0 visits vs 28.4% for $\ge 2$ visits) and medication changes (13.1% for active titration vs 9.8% for unchanged regimens).

*Observational Finding Note: All empirical relationships reflect statistical correlations that guide operational review prioritization; they are not treated as clinical or causal conclusions.*

---

## 🤖 Machine Learning

Four candidate models were evaluated using a standardized scikit-learn pipeline, with preprocessing fitted strictly on training data:
1. **Majority Class Baseline**: Dummy classifier predicting the negative majority class.
2. **Logistic Regression Baseline**: $L_2$-regularized linear model with balanced class weights (`C=1.0`).
3. **Random Forest Classifier**: Non-linear ensemble (`n_estimators=150`, `max_depth=12`, `min_samples_leaf=10`, `class_weight='balanced_subsample'`).
4. **Calibrated Random Forest**: Platt sigmoid scaling via `CalibratedClassifierCV` to align predicted probabilities with empirical event rates.

### Held-Out Test Set Performance (20,011 Encounters, Zero Patient Overlap)

| Model | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| Majority Class Baseline | 0.5000 | 0.1139 | 0.1139 |
| Logistic Regression | 0.6453 | 0.1992 | 0.2284 |
| Random Forest | 0.6576 | 0.2030 | 0.2047 |
| **Calibrated Random Forest** | **0.6572** | **0.2054** | **0.0973** |

*Key Takeaway: Calibrated Random Forest achieved the highest Average Precision (PR-AUC: 0.2054) and reduced probability calibration error by 52.5% (Brier score: 0.0973 vs 0.2047 uncalibrated).*

---

## 🎯 Operational Risk Prioritization

Under an 11.34% base event rate, static classification thresholds (e.g. 0.50) fail to guide daily hospital staffing. Instead, the platform dynamically ranks patients by calibrated risk and cuts off at the hospital's available review capacity:

| Review Capacity | Prioritized Encounters | Cutoff Risk Threshold | Captured Readmissions | Operational Lift | Precision at Capacity |
|:---|---:|---:|---:|---:|---:|
| **Top 5% Capacity** | 1,001 reviews | $\ge 22.97\%$ | 293 / 2,280 | **2.57×** | 29.27% |
| **Top 10% Capacity** | 2,002 reviews | $\ge 19.36\%$ | 484 / 2,280 | **2.12×** | 24.18% |
| **Top 20% Capacity** | 4,003 reviews | $\ge 15.21\%$ | 847 / 2,280 | **1.86×** | 21.16% |

Each prioritized discharge is automatically routed to an administrative workflow:
- *Care-Management Review*: Multi-condition complexity and high acute utilization.
- *Discharge-Planning Review*: Facility transfers and prolonged hospital stays ($\ge 7$ days).
- *Follow-Up Coordination Review*: Active diabetes medication titration during admission.
- *Additional Record Review*: Sparse history requiring chart audit.

*These queues establish administrative review priorities only; they do not dictate clinical treatments.*

---

## 🔎 Explainability & Responsible AI

- **TreeSHAP Attributions**: Global factor attribution identifies historical inpatient frequency (`prior_inpatient_ratio`, `number_inpatient`), discharge destination (`discharge_disp_clean_Home` vs `Facility_Transfer`), and chronic utilization flags (`high_acute_utilizer`) as primary model drivers.
- **Demographic Fairness Monitoring**: Continuous parity tracking across Race (selection rates: 9.8%–11.2%), Gender (Female: 10.0% vs Male: 9.9%), and Age groups confirms the operational triage queue operates without systematic demographic selection disparity.
- **LLM Analytics Assistant**: Context-grounded conversational agent powered by structured, pre-verified Python calculations. Synthesizes executive briefings with zero statistical hallucination.
- **Safe & Secure**: Operates in full offline mode without requiring external credentials; zero hard-coded API keys; zero patient-specific diagnostic or treatment prescriptions.

---

## 🖥️ Streamlit Dashboard

The platform features a 9-page interactive decision-support application:

1. **Executive Overview**: High-level operational intelligence, core cohort KPIs, target distribution, and benchmark summary.
2. **Data Quality**: Detailed missingness audit, zero-variance screening, and cohort eligibility filtering logic.
3. **Cohort & Encounter Analytics**: Demographic breakdowns, utilization distributions, and diagnostic complexity.
4. **Readmission Analysis**: Observational associations across prior utilization, medication changes, and admission types.
5. **Predictive Modeling**: Held-out test set benchmarks, internal CV results, ROC curve, PR curve, and calibration reliability diagram.
6. **Risk Prioritization**: Interactive hospital review capacity slider (1%–30%), preset queue thresholds, workflow badges, and CSV queue export.
7. **SHAP Explainability**: Global feature impact rankings and encounter-level factor attributions.
8. **LLM Analytics Assistant**: Natural language inquiry grounded strictly in verified Python metrics (offline briefs or dynamic OpenAI API).
9. **About / Methodology**: Data provenance, zero-leakage protocols, demographic fairness audit tables, and AI governance.

---

## 🛠️ Tech Stack

`Python 3.13` • `Pandas` • `NumPy` • `Scikit-Learn` • `Streamlit` • `SHAP` • `Matplotlib` • `Seaborn` • `Pytest` • `OpenAI API`

---

## ▶️ Run Locally

### Prerequisites
- Python 3.10+ (Python 3.13 recommended)
- Git

### Installation & Execution

```powershell
# 1. Clone repository
git clone https://github.com/mohitjoshi-dev/hospital-readmission-intelligence.git
cd hospital-readmission-intelligence

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Linux/macOS: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite
pytest tests/ -v

# 5. Launch interactive Streamlit dashboard
streamlit run dashboard/app.py
```

Access the dashboard at **[http://localhost:8501](http://localhost:8501)**.  
*(Optional: Add `OPENAI_API_KEY=sk-...` to `.env` to enable dynamic LLM chat; offline analytical briefs work automatically out of the box).*

---

## ⚠️ Limitations

- **Historical EHR Data (1999–2008)**: Clinical protocols, hospital admission guidelines, and diabetes pharmacotherapy (e.g. SGLT2 inhibitors, GLP-1 receptor agonists) have advanced substantially since 2008.
- **Observational Correlation**: Identified risk factors reflect statistical associations that guide operational review priority, not biological causes of readmission.
- **Dataset-Specific Performance**: Metrics reflect care patterns across the 130 participating hospitals; external validation on contemporary institutional cohorts is recommended prior to clinical deployment.
- **Non-Clinical Boundaries**: The platform is strictly an operational triage tool and does not provide medical decision-making or patient-specific treatment plans.

---

## 📚 References

- **UCI Dataset Repository**: [Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)
- **Original Study**: Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Balasubramanian, S., Higgins, K. A., & Cios, K. J. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International, vol. 2014, Article ID 781670.

---

## 👨💻 Author

**Mohit Joshi**  
GitHub: [https://github.com/mohitjoshi-dev](https://github.com/mohitjoshi-dev)
