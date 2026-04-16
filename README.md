# Pima Indian Diabetes Analysis

A full-stack healthtech data analysis project built with **Django 6**, **PostgreSQL 18**, and **scikit-learn** — using the [Pima Indian Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (768 patients, 9 features).

## What's Inside

| Feature | Description |
|---|---|
| **Patient Explorer** | Browse, filter, and search all 768 patients stored in PostgreSQL |
| **Statistics Dashboard** | Aggregated metrics: averages, prevalence rates, age distribution |
| **Charts** | 5 interactive Chart.js visualizations powered by a live JSON API |
| **ML Risk Predictor** | Random Forest model (AUC 0.818) with per-patient SHAP explanations |
| **Ask AI** | Natural language queries powered by Groq (LLaMA 3.3 70B) → text-to-SQL |
| **Jupyter EDA** | Full exploratory analysis notebook with K-Means clustering + SHAP |
| **Django Admin** | Auto-generated GUI to browse and edit all patient records |

## Tech Stack

- **Backend:** Django 6, Python 3.13
- **Database:** PostgreSQL 18 (via psycopg2)
- **ML:** scikit-learn (Logistic Regression, Random Forest), SHAP, K-Means
- **Data Analysis:** pandas, seaborn, matplotlib, SQLAlchemy
- **Frontend:** Bootstrap 5, Chart.js 4 (CDN)
- **AI:** Groq API (LLaMA 3.3 70B) for natural language → SQL

## Setup

### 1. Prerequisites
- Python 3.11+
- PostgreSQL running locally

### 2. Clone and install
```bash
git clone https://github.com/ramjhawar-alt/pima-indian-diabetes-analysis.git
cd pima-indian-diabetes-analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Create the database
```bash
psql postgres -c "CREATE DATABASE diabetes_db;"
```

### 4. Configure settings
Edit `pima_project/settings.py` — update the `DATABASES` user to your local PostgreSQL username:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'diabetes_db',
        'USER': 'your_username',   # change this
        'PASSWORD': '',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

### 5. Run migrations and load data
```bash
python manage.py migrate
python manage.py load_csv          # imports all 768 patients from diabetes.csv
```

### 6. Train the ML model
```bash
python manage.py train_model       # trains Random Forest + Logistic Regression, saves best
```

### 7. Create admin user
```bash
python manage.py createsuperuser
```

### 8. Start the server
```bash
python manage.py runserver
```

Open `http://127.0.0.1:8000`

## Pages

| URL | Description |
|---|---|
| `/patients/` | Patient list with filters |
| `/patients/<id>/` | Single patient detail |
| `/stats/` | Population statistics |
| `/charts/` | Interactive visualizations |
| `/predict/` | Diabetes risk predictor |
| `/ask/` | Natural language AI query |
| `/admin/` | Django admin panel |

## Jupyter Notebook

```bash
cd analysis
jupyter notebook eda.ipynb
```

The notebook connects directly to PostgreSQL and produces:
- Distribution plots, box plots, correlation heatmap
- BMI vs Glucose scatter plot
- K-Means patient segmentation (elbow plot + cluster analysis)
- SHAP summary, waterfall, and dependence plots

## ML Model Performance

| Model | Cross-Val AUC | Test Accuracy | Test AUC |
|---|---|---|---|
| Logistic Regression | 0.843 | 71% | 0.813 |
| **Random Forest** | 0.820 | **75%** | **0.818** |

Top predictors (SHAP): Glucose (26.8%), BMI (15.8%), Diabetes Pedigree (12.5%), Age (12.0%)
