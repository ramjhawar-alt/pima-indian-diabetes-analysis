# Pima Indian Diabetes — Full-Stack Analysis

![Django](https://img.shields.io/badge/Django-6.0-092E20?logo=django&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?logo=postgresql&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)

Public portfolio project: a **Django + PostgreSQL** web application and **Jupyter** analysis pipeline around the classic [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) benchmark (768 records, clinical features + outcome). It demonstrates **ORM-backed data modeling**, **REST-style JSON APIs for charts**, **ML deployment in-app** (Random Forest + SHAP), and **LLM-assisted analytics** (natural language → read-only SQL via Groq).

---

## Highlights

- **End-to-end web stack** — Models, migrations, admin, class-based data access, templates with Bootstrap 5.
- **Interactive analytics** — Chart.js dashboards fed by Django views aggregating live PostgreSQL data.
- **Responsible ML surface** — Saved `scikit-learn` pipeline, probability + risk tiering, **SHAP** explanations on the prediction page (aligned with notebook analysis).
- **AI-assisted querying** — `/ask/` converts plain English to `SELECT`-only SQL against the documented schema (Groq / LLaMA 3.3 70B).
- **Reproducible research** — `analysis/eda.ipynb`: distributions, correlations, K-Means segmentation, SHAP global and dependence plots.

---

## Architecture (at a glance)

| Layer | Role |
|--------|------|
| **PostgreSQL** | Source of truth for patient rows; notebook connects via SQLAlchemy. |
| **Django (`patients`)** | CRUD-style pages, stats, chart JSON endpoints, prediction + SHAP, NL→SQL orchestration. |
| **Static ML artifact** | `train_model` management command writes `patients/ml/diabetes_model.joblib` (gitignored; train after clone). |
| **Notebook** | EDA, clustering, SHAP — same DB and model path as the app for consistency. |

---

## Feature map

| Capability | What it shows |
|------------|----------------|
| Patient explorer | Filtering, search, detail views over ORM data. |
| Statistics | Aggregations and prevalence-style summaries. |
| Charts | Five Chart.js views backed by `/charts/api/`. |
| Predict + SHAP | Random Forest risk score with per-request explanation. |
| Ask AI | Natural language questions → generated SQL → tabular results (read-only). |
| Admin | Django admin for inspection and edits. |

---

## Routes

| Path | Purpose |
|------|---------|
| `/` | Entry / redirect to patient list |
| `/patients/` | List + filters |
| `/patients/<id>/` | Detail |
| `/stats/` | Dashboard metrics |
| `/charts/` | Interactive charts |
| `/predict/` | ML + SHAP |
| `/ask/` | NL → SQL |
| `/admin/` | Admin site |

---

## Local setup

**Requirements:** Python 3.11+, PostgreSQL, a [Groq](https://console.groq.com) API key for `/ask/`.

```bash
git clone https://github.com/ramjhawar-alt/pima-indian-diabetes-analysis.git
cd pima-indian-diabetes-analysis
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Database:** create an empty database (name should match `pima_project/settings.py`, default `diabetes_db`):

```bash
createdb diabetes_db
# or: psql -U postgres -c "CREATE DATABASE diabetes_db;"
```

**Configure `DATABASES` in `pima_project/settings.py`** — set `USER` (and `PASSWORD` if needed) to your PostgreSQL role. Optionally use environment-driven settings in a fork for production-style config.

**Environment (do not commit secrets):**

```bash
export GROQ_API_KEY="your_key_here"
```

See `.env.example` for the variable name; load with your shell or a tool like `direnv` if you prefer.

**Initialize data and model:**

```bash
python manage.py migrate
python manage.py load_csv
python manage.py train_model
python manage.py createsuperuser   # optional, for /admin/
python manage.py runserver
```

Then open **http://127.0.0.1:8000**. The **Ask AI** page requires `GROQ_API_KEY` to be set; other pages work without it once the DB is loaded.

---

## Jupyter notebook

```bash
cd analysis
jupyter notebook eda.ipynb
```

The notebook expects PostgreSQL connectivity consistent with your `settings.py` / SQLAlchemy URI. It covers data quality (zero-as-missing handling), distributions, correlation, K-Means (elbow + cluster interpretation), and SHAP plots using the trained Random Forest.

---

## Model snapshot (reference)

Reported on held-out test data from the bundled training command:

| Model | Cross-val AUC | Test accuracy | Test AUC |
|--------|----------------|-----------------|----------|
| Logistic Regression | 0.843 | 71% | 0.813 |
| **Random Forest** | 0.820 | **75%** | **0.818** |

SHAP-style global emphasis (Random Forest): glucose, BMI, diabetes pedigree function, age among top contributors — useful for discussion in interviews, not clinical use.

---

## Data & disclaimer

This repository uses the **public UCI / Kaggle Pima Indians Diabetes** dataset for **education and portfolio demonstration only**. It is **not** a medical device, diagnostic tool, or substitute for professional care.

---

## Acknowledgments

- Dataset: Smith *et al.*, *Journal of the American Medical Informatics Association* (1998); widely mirrored on [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
- UI: [Bootstrap](https://getbootstrap.com/), [Chart.js](https://www.chartjs.org/).
- LLM inference: [Groq](https://groq.com/).
