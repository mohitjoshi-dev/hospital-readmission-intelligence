# Raw Healthcare Dataset Directory

This directory is designated for the raw, immutable clinical dataset files:
- `diabetic_data.csv`
- `IDS_mapping.csv`

## Dataset Provenance & Source
The dataset is obtained separately from the **UCI Machine Learning Repository**:
- **Dataset Title**: Diabetes 130-US hospitals for years 1999-2008 Data Set
- **Publication**: Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International, 2014.
- **Repository URL**: [UCI Machine Learning Repository - Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

## Local Placement
Download the original archive from the source above and extract the two files directly into this directory:
```
data/raw/
├── diabetic_data.csv
└── IDS_mapping.csv
```
*(Alternatively, place them in `raw/` at the project root).*

## GitHub Exclusion Notice
Raw clinical healthcare datasets contain patient-encounter records and are **intentionally excluded from the Git repository** via `.gitignore` in accordance with healthcare data governance, privacy standards, and repository size best practices.
