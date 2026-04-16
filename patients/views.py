"""
Views are the heart of a Django web app.

A view is just a Python function (or class) that:
  1. Receives an HttpRequest object (the browser's request)
  2. Does some work — usually a database query
  3. Returns an HttpResponse (usually rendered HTML)

The flow is always:   URL  →  view function  →  template  →  HTML back to browser
"""

import json
import re

from django.conf import settings
from django.db import connection
from django.db.models import Avg, Count, Max, Min, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render

from .models import Patient

# Schema description sent to the AI so it understands the database structure
SCHEMA_DESCRIPTION = """
You are a SQL expert. The PostgreSQL database has one table called "patients_patient" with these columns:
  - id          INTEGER  (auto-generated primary key)
  - pregnancies INTEGER  (number of pregnancies)
  - glucose     FLOAT    (plasma glucose concentration in mg/dL)
  - blood_pressure FLOAT (diastolic blood pressure in mm Hg)
  - skin_thickness FLOAT (triceps skin fold thickness in mm)
  - insulin     FLOAT    (2-hour serum insulin in μU/mL)
  - bmi         FLOAT    (body mass index)
  - diabetes_pedigree FLOAT (genetic diabetes risk score, higher = more family history)
  - age         INTEGER  (age in years)
  - outcome     BOOLEAN  (TRUE = patient has diabetes, FALSE = patient does not)

The table has 768 rows of data about Pima Indian women.

Rules:
- Only write SELECT queries. Never write INSERT, UPDATE, DELETE, DROP, or any other mutating SQL.
- Always reference the table as "patients_patient".
- Return ONLY the raw SQL query with no explanation, no markdown, no code fences, no backticks.
- End the query with a semicolon.
- When the question asks about "diabetic" patients, use WHERE outcome = true.
- When the question asks about "non-diabetic", use WHERE outcome = false.
"""


def patient_list(request):
    """
    GET /patients/

    Shows all patients in a table with optional filtering.

    request.GET is a dict of URL query parameters, e.g.:
        /patients/?outcome=1      → only diabetic patients
        /patients/?min_age=40     → patients aged 40+
        /patients/?outcome=0&min_age=30

    Django's ORM filter() method translates these into SQL WHERE clauses.
    Chaining .filter() calls adds AND conditions.
    """
    queryset = Patient.objects.all()

    # --- Filter: outcome (diabetic / non-diabetic / all) ---
    outcome_filter = request.GET.get("outcome", "")
    if outcome_filter == "1":
        queryset = queryset.filter(outcome=True)
    elif outcome_filter == "0":
        queryset = queryset.filter(outcome=False)

    # --- Filter: minimum age ---
    min_age = request.GET.get("min_age", "")
    if min_age.isdigit():
        queryset = queryset.filter(age__gte=int(min_age))

    # --- Filter: maximum age ---
    max_age = request.GET.get("max_age", "")
    if max_age.isdigit():
        queryset = queryset.filter(age__lte=int(max_age))

    # --- Filter: minimum glucose ---
    min_glucose = request.GET.get("min_glucose", "")
    if min_glucose.replace(".", "", 1).isdigit():
        queryset = queryset.filter(glucose__gte=float(min_glucose))

    total_count = queryset.count()

    # render() takes the request, a template path, and a context dict.
    # The context dict is what becomes available inside the HTML template.
    context = {
        "patients": queryset,
        "total_count": total_count,
        "outcome_filter": outcome_filter,
        "min_age": min_age,
        "max_age": max_age,
        "min_glucose": min_glucose,
    }
    return render(request, "patients/list.html", context)


def patient_detail(request, pk):
    """
    GET /patients/<pk>/

    Shows all fields for a single patient.

    get_object_or_404() is a Django shortcut:
      - If the patient exists → return it
      - If not → automatically return a 404 page instead of crashing

    'pk' stands for Primary Key — the auto-generated integer id Django adds to every model.
    """
    patient = get_object_or_404(Patient, pk=pk)
    context = {"patient": patient}
    return render(request, "patients/detail.html", context)


def stats(request):
    """
    GET /stats/

    Uses Django's aggregation functions to compute summary statistics directly
    in PostgreSQL (not in Python) — so only one fast SQL query runs, regardless
    of how many rows there are.

    aggregate() returns a plain Python dict, e.g.:
      {"glucose__avg": 120.9, "age__avg": 33.2, ...}

    annotate() adds per-group statistics (here: split by outcome).
    """

    # Overall statistics across all 768 patients
    overall = Patient.objects.aggregate(
        avg_glucose=Avg("glucose"),
        avg_bmi=Avg("bmi"),
        avg_age=Avg("age"),
        avg_blood_pressure=Avg("blood_pressure"),
        avg_insulin=Avg("insulin"),
        avg_pregnancies=Avg("pregnancies"),
        min_age=Min("age"),
        max_age=Max("age"),
        total=Count("id"),
        diabetic_count=Count("id", filter=Q(outcome=True)),
    )

    # Statistics broken down by outcome (True vs False)
    # values("outcome") groups the queryset — like SQL GROUP BY
    by_outcome = (
        Patient.objects.values("outcome")
        .annotate(
            count=Count("id"),
            avg_glucose=Avg("glucose"),
            avg_bmi=Avg("bmi"),
            avg_age=Avg("age"),
            avg_blood_pressure=Avg("blood_pressure"),
        )
        .order_by("outcome")
    )

    # Age distribution buckets for a visual breakdown
    age_buckets = [
        {"label": "21–30", "count": Patient.objects.filter(age__gte=21, age__lte=30).count()},
        {"label": "31–40", "count": Patient.objects.filter(age__gte=31, age__lte=40).count()},
        {"label": "41–50", "count": Patient.objects.filter(age__gte=41, age__lte=50).count()},
        {"label": "51–60", "count": Patient.objects.filter(age__gte=51, age__lte=60).count()},
        {"label": "61+",   "count": Patient.objects.filter(age__gte=61).count()},
    ]

    context = {
        "overall": overall,
        "by_outcome": by_outcome,
        "age_buckets": age_buckets,
        "diabetic_pct": round(overall["diabetic_count"] / overall["total"] * 100, 1),
    }
    return render(request, "patients/stats.html", context)


def ask(request):
    """
    GET  /ask/  — shows the empty question form
    POST /ask/  — sends the question to Groq, gets SQL back, runs it, shows results

    The flow:
      1. User types a plain English question
      2. We send the question + database schema to Groq (LLaMA 3.3 70B model)
      3. Groq returns a SQL SELECT query
      4. We execute that query against PostgreSQL using Django's raw DB cursor
      5. We render the results in a table

    request.method lets us handle GET (show form) and POST (process question)
    in the same view function.
    """
    question = ""
    sql_query = ""
    columns = []
    rows = []
    error = ""

    if request.method == "POST":
        question = request.POST.get("question", "").strip()

        if question:
            try:
                from groq import Groq

                client = Groq(api_key=settings.GROQ_API_KEY)

                # Send the question to Groq with the schema context
                chat = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": SCHEMA_DESCRIPTION},
                        {"role": "user",   "content": question},
                    ],
                    temperature=0,       # 0 = most deterministic, best for SQL
                    max_tokens=512,
                )

                sql_query = chat.choices[0].message.content.strip()

                # Strip markdown code fences if the model added them anyway
                sql_query = re.sub(r"^```[a-z]*\n?", "", sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r"\n?```$", "", sql_query)
                sql_query = sql_query.strip()

                # Safety: only allow SELECT statements
                first_word = sql_query.split()[0].upper() if sql_query else ""
                if first_word != "SELECT":
                    error = "The AI returned a non-SELECT query. Only SELECT queries are allowed."
                else:
                    # Execute the SQL directly against PostgreSQL
                    # connection.cursor() gives us a raw database cursor —
                    # useful when you need to run SQL that the ORM can't express
                    with connection.cursor() as cursor:
                        cursor.execute(sql_query)
                        columns = [desc[0] for desc in cursor.description]
                        rows = cursor.fetchall()

            except Exception as exc:
                error = str(exc)

    context = {
        "question": question,
        "sql_query": sql_query,
        "columns": columns,
        "rows": rows,
        "error": error,
        "row_count": len(rows),
    }
    return render(request, "patients/ask.html", context)


def charts(request):
    """
    GET /charts/

    Renders the charts dashboard page. The actual chart data is fetched
    separately by the browser via the /charts/api/ endpoint (see below).
    This separation is the standard pattern for dashboards:
      - The page loads instantly (no data queries on page load)
      - JavaScript then calls the API and renders the charts asynchronously
    """
    return render(request, "patients/charts.html")


def charts_api(request):
    """
    GET /charts/api/

    Returns all chart data as JSON. Called by Chart.js in the browser.

    JsonResponse converts a Python dict → JSON HTTP response automatically.
    This is how REST APIs work: the backend speaks JSON, the frontend renders it.
    """
    total = Patient.objects.count()
    diabetic_count = Patient.objects.filter(outcome=True).count()
    non_diabetic_count = total - diabetic_count

    # 1. Outcome doughnut chart data
    outcome_data = {
        "labels": ["Diabetic", "Non-diabetic"],
        "counts": [diabetic_count, non_diabetic_count],
    }

    # 2. Average metrics comparison: diabetic vs non-diabetic
    def group_avgs(outcome_val):
        return Patient.objects.filter(outcome=outcome_val).aggregate(
            avg_glucose=Avg("glucose"),
            avg_bmi=Avg("bmi"),
            avg_age=Avg("age"),
            avg_blood_pressure=Avg("blood_pressure"),
            avg_pregnancies=Avg("pregnancies"),
        )

    diabetic_avgs = group_avgs(True)
    non_diabetic_avgs = group_avgs(False)

    comparison_data = {
        "labels": ["Glucose (mg/dL)", "BMI", "Age (years)", "Blood Pressure", "Pregnancies"],
        "diabetic": [
            round(diabetic_avgs["avg_glucose"] or 0, 1),
            round(diabetic_avgs["avg_bmi"] or 0, 1),
            round(diabetic_avgs["avg_age"] or 0, 1),
            round(diabetic_avgs["avg_blood_pressure"] or 0, 1),
            round(diabetic_avgs["avg_pregnancies"] or 0, 1),
        ],
        "non_diabetic": [
            round(non_diabetic_avgs["avg_glucose"] or 0, 1),
            round(non_diabetic_avgs["avg_bmi"] or 0, 1),
            round(non_diabetic_avgs["avg_age"] or 0, 1),
            round(non_diabetic_avgs["avg_blood_pressure"] or 0, 1),
            round(non_diabetic_avgs["avg_pregnancies"] or 0, 1),
        ],
    }

    # 3. Glucose distribution histogram (20 buckets from 0 to 200)
    # We use raw SQL here because Django's ORM doesn't have a built-in histogram
    bucket_size = 10
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT
                FLOOR(glucose / %s) * %s AS bucket_start,
                COUNT(*) FILTER (WHERE outcome = false) AS non_diabetic,
                COUNT(*) FILTER (WHERE outcome = true)  AS diabetic
            FROM patients_patient
            WHERE glucose > 0
            GROUP BY bucket_start
            ORDER BY bucket_start
        """, [bucket_size, bucket_size])
        rows = cursor.fetchall()

    glucose_data = {
        "labels": [f"{int(r[0])}-{int(r[0])+bucket_size}" for r in rows],
        "non_diabetic": [r[1] for r in rows],
        "diabetic":     [r[2] for r in rows],
    }

    # 4. Age distribution by decade, split by outcome
    age_ranges = [(21, 30), (31, 40), (41, 50), (51, 60), (61, 100)]
    age_labels = ["21-30", "31-40", "41-50", "51-60", "61+"]
    age_diabetic, age_non_diabetic = [], []
    for lo, hi in age_ranges:
        age_diabetic.append(Patient.objects.filter(age__gte=lo, age__lte=hi, outcome=True).count())
        age_non_diabetic.append(Patient.objects.filter(age__gte=lo, age__lte=hi, outcome=False).count())

    age_data = {
        "labels": age_labels,
        "diabetic": age_diabetic,
        "non_diabetic": age_non_diabetic,
    }

    # 5. Scatter plot: all patients as {x: glucose, y: bmi, outcome: bool}
    # Limit to 768 points (small enough to send to the browser as JSON)
    scatter_points = list(
        Patient.objects.values("glucose", "bmi", "outcome")
                       .filter(glucose__gt=0, bmi__gt=0)
    )
    # Convert boolean to string for Chart.js dataset separation
    scatter_data = {
        "diabetic":     [{"x": p["glucose"], "y": round(p["bmi"], 1)} for p in scatter_points if p["outcome"]],
        "non_diabetic": [{"x": p["glucose"], "y": round(p["bmi"], 1)} for p in scatter_points if not p["outcome"]],
    }

    return JsonResponse({
        "outcome": outcome_data,
        "comparison": comparison_data,
        "glucose": glucose_data,
        "age": age_data,
        "scatter": scatter_data,
    })


# Load the trained model once at module level so it isn't re-loaded on every request.
# _MODEL is None until the first request that needs it (lazy loading).
_MODEL_CACHE = None

def _load_model():
    """Load the saved joblib model; returns None if not yet trained."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    import os
    import joblib
    model_path = os.path.join(
        settings.BASE_DIR, "patients", "ml", "diabetes_model.joblib"
    )
    if os.path.exists(model_path):
        _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE


def predict(request):
    """
    GET  /predict/  — shows the input form with default/average values
    POST /predict/  — runs the saved ML model, returns risk probability

    The model was trained with scikit-learn and saved as a Pipeline object.
    pipeline.predict_proba(X) returns [[p_no_diabetes, p_diabetes]] for each row.
    We take p_diabetes (index 1) and display it as a percentage risk score.
    """
    import numpy as np

    # Default values pre-filled in the form — the dataset averages
    defaults = {
        "pregnancies": 3,
        "glucose": 121,
        "blood_pressure": 69,
        "skin_thickness": 21,
        "insulin": 80,
        "bmi": 32.0,
        "diabetes_pedigree": 0.47,
        "age": 33,
    }

    prediction = None
    probability = None
    risk_level = None
    risk_color = None
    feature_contributions = None
    error = None

    if request.method == "POST":
        model_data = _load_model()
        if model_data is None:
            error = "Model not trained yet. Run: python manage.py train_model"
        else:
            try:
                pipeline = model_data["pipeline"]
                feature_cols = model_data["feature_cols"]

                # Read form values, falling back to defaults if missing/invalid
                values = {}
                for col in feature_cols:
                    raw = request.POST.get(col, "")
                    try:
                        values[col] = float(raw)
                    except (ValueError, TypeError):
                        values[col] = float(defaults.get(col, 0))

                # Build a (1 x 8) NumPy array in the correct column order
                X = np.array([[values[col] for col in feature_cols]])

                # predict_proba returns [[prob_class_0, prob_class_1]]
                proba = pipeline.predict_proba(X)[0]
                probability = round(float(proba[1]) * 100, 1)
                prediction = int(pipeline.predict(X)[0])

                # Determine risk level for display
                if probability < 30:
                    risk_level, risk_color = "Low Risk", "success"
                elif probability < 60:
                    risk_level, risk_color = "Medium Risk", "warning"
                else:
                    risk_level, risk_color = "High Risk", "danger"

                # SHAP: per-patient explanation of why the model predicted this score
                rf_model = pipeline.named_steps.get("model")
                imputer  = pipeline.named_steps.get("imputer")
                if rf_model is not None and imputer is not None:
                    try:
                        import shap as shap_lib
                        X_imp = imputer.transform(X)
                        explainer   = shap_lib.TreeExplainer(rf_model)
                        explanation = explainer(X_imp, check_additivity=False)
                        # explanation.values shape: (1, n_features, n_classes)
                        shap_row = explanation.values[0, :, 1]
                        # Build list of (label, raw_value, shap_value, pct_of_max)
                        max_abs = max(abs(v) for v in shap_row) or 1
                        feature_contributions = sorted(
                            [
                                {
                                    "label": f.replace("_", " ").title(),
                                    "value": round(values[f], 2),
                                    "shap": round(float(sv), 4),
                                    "bar_pct": round(abs(float(sv)) / max_abs * 100, 1),
                                    "direction": "up" if sv > 0 else "down",
                                }
                                for f, sv in zip(feature_cols, shap_row)
                            ],
                            key=lambda x: -abs(x["shap"])
                        )
                    except Exception:
                        # Fall back to static feature importances if SHAP fails
                        importances = rf_model.feature_importances_
                        feature_contributions = sorted(
                            [
                                {
                                    "label": f.replace("_", " ").title(),
                                    "value": round(values[f], 2),
                                    "shap": round(float(imp), 4),
                                    "bar_pct": round(float(imp) * 100, 1),
                                    "direction": "up",
                                }
                                for f, imp in zip(feature_cols, importances)
                            ],
                            key=lambda x: -x["bar_pct"]
                        )

                # Keep entered values in the form
                defaults = values

            except Exception as exc:
                error = str(exc)

    context = {
        "defaults": defaults,
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "feature_contributions": feature_contributions,
        "error": error,
        "model_info": _load_model(),
    }
    return render(request, "patients/predict.html", context)
