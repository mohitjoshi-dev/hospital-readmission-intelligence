# Hospital Readmission Intelligence

An operational healthcare analytics and decision-support platform that estimates 30-day readmission risk and prioritizes hospital discharge reviews under constrained staff capacity.

> **Disclaimer**: This platform is an operational decision-support demonstration using historical EHR data. It is not a clinical diagnostic, treatment, or medical decision-making system.

---

### 🚀 Local Demo
Run the application locally and access the interactive dashboard at:  
👉 **[http://localhost:8501](http://localhost:8501)**

---

### 📊 Dataset
- **Dataset**: Diabetes 130-US Hospitals for Years 1999–2008 (100,111 eligible inpatient encounters across 70,436 unique patients).
- **Official Source**: [UCI Machine Learning Repository — Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)
- **Raw Data Note**: Raw CSV files are intentionally excluded from Git in accordance with healthcare data governance and repository size standards. Download the dataset from the official source above and place the files in `data/raw/`.

---

### ⚡ What This Project Does
- **Capacity-Based Prioritization**: Ranks hospitalized patients into dynamic operational review queues (Top 5%, 10%, 20%), achieving a **2.12x operational lift** over random selection.
- **Administrative Workflow Routing**: Categorizes high-risk discharges into administrative review queues (*Care-Management Review*, *Discharge-Planning Review*, *Follow-Up Coordination Review*, *Additional Record Review*).
- **Leak-Free Predictive Modeling**: Patient-grouped 80/20 train/test evaluation (zero patient overlap) with probability calibration (Brier score: 0.097).
- **Model Explainability**: Uses TreeSHAP for global and patient-level operational factor attributions.
- **Demographic Fairness Monitoring**: Continuously tracks review selection parity across Race, Gender, and Age cohorts.
- **LLM Analytics Assistant**: Context-grounded conversational agent providing executive analytical briefs with zero statistical hallucination (functions offline or with OpenAI API).

---

### 🛠️ Tech Stack
`Python 3.13` • `Scikit-Learn` • `Streamlit` • `SHAP` • `Pandas` • `NumPy` • `Matplotlib` • `Seaborn` • `Pytest` • `OpenAI API`

---

### 💻 Run Locally

```powershell
# 1. Clone repository and navigate to project root
git clone https://github.com/mohitjoshi-dev/Hospital-Readmission-Intelligence.git
cd Hospital-Readmission-Intelligence

# 2. Set up virtual environment and install dependencies
py -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# 3. Launch the interactive dashboard
.\.venv\Scripts\streamlit run dashboard/app.py
```
*(Optional: Add `OPENAI_API_KEY` to `.env` to enable dynamic LLM chat; verified offline briefs work out of the box).*

---

### 👤 Author
**Mohit Joshi**  
GitHub: [https://github.com/mohitjoshi-dev](https://github.com/mohitjoshi-dev)
