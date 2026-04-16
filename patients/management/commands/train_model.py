"""
Custom Django management command: python manage.py train_model

This script:
  1. Loads all 768 patients from PostgreSQL into a pandas DataFrame
  2. Handles missing values (zeros that are biologically impossible)
  3. Trains two models: Logistic Regression and Random Forest
  4. Evaluates both with accuracy, precision, recall, F1, and a confusion matrix
  5. Saves the better model to patients/ml/diabetes_model.joblib

Run with:
    python manage.py train_model

In healthtech, this is what a data scientist would run once (or on a schedule)
to rebuild the prediction model whenever new patient data arrives.
"""

import os

import joblib
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from patients.models import Patient

# Path where the trained model will be saved
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ml",
    "diabetes_model.joblib",
)

# The 8 features the model uses as inputs — same as the dataset columns
FEATURE_COLS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]


class Command(BaseCommand):
    help = "Train Logistic Regression and Random Forest models on the patient data, save the best one"

    def handle(self, *args, **options):
        # scikit-learn imports are here (not top-level) so Django loads faster
        # when it's not needed — standard practice for heavy imports
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # ── 1. Load data from PostgreSQL ─────────────────────────────────────────
        self.stdout.write("Loading patient data from PostgreSQL…")
        qs = Patient.objects.values(*FEATURE_COLS, "outcome")
        df = pd.DataFrame(list(qs))

        self.stdout.write(f"  {len(df)} patients loaded.")

        # ── 2. Handle missing values (zeros that are biologically impossible) ────
        # From the EDA, we know these columns cannot be zero in reality
        zero_as_missing = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
        df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

        missing_counts = df[zero_as_missing].isnull().sum()
        self.stdout.write("Missing values (zeros replaced with NaN):")
        for col, count in missing_counts[missing_counts > 0].items():
            self.stdout.write(f"  {col}: {count} ({count/len(df)*100:.1f}%)")

        # ── 3. Prepare features (X) and target (y) ────────────────────────────────
        X = df[FEATURE_COLS].values.astype(float)
        y = df["outcome"].astype(int).values

        self.stdout.write(f"\nClass balance: {y.sum()} diabetic ({y.mean()*100:.1f}%), "
                         f"{(y==0).sum()} non-diabetic ({(1-y.mean())*100:.1f}%)")

        # ── 4. Train/test split ───────────────────────────────────────────────────
        # stratify=y ensures the train and test sets have the same class ratio
        # test_size=0.2 means 80% training, 20% testing (standard split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.stdout.write(f"  Train: {len(X_train)} patients | Test: {len(X_test)} patients")

        # ── 5. Build pipelines ────────────────────────────────────────────────────
        # A Pipeline chains preprocessing + model into one object.
        # This is important: the imputer and scaler are fit ONLY on training data,
        # then applied to test data — preventing "data leakage".
        #
        # SimpleImputer: fills NaN with the median of that column
        # StandardScaler: rescales all features to mean=0, std=1 (needed for LogReg)

        lr_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(max_iter=1000, random_state=42)),
        ])

        rf_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ])

        # ── 6. Cross-validation ───────────────────────────────────────────────────
        # Cross-validation splits the training set 5 ways, trains on 4 folds,
        # tests on 1 fold — repeated 5 times. Gives a more reliable accuracy
        # estimate than a single train/test split.
        self.stdout.write("\n── Cross-Validation (5-fold, training set only) ──")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, pipeline in [("Logistic Regression", lr_pipeline), ("Random Forest", rf_pipeline)]:
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
            self.stdout.write(
                f"  {name}: AUC = {scores.mean():.3f} ± {scores.std():.3f}"
            )

        # ── 7. Final training + evaluation on held-out test set ───────────────────
        self.stdout.write("\n── Final Evaluation on Test Set ──")
        results = {}

        for name, pipeline in [("Logistic Regression", lr_pipeline), ("Random Forest", rf_pipeline)]:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            results[name] = {"pipeline": pipeline, "auc": auc}

            self.stdout.write(f"\n{name}:")
            self.stdout.write(f"  ROC-AUC:  {auc:.3f}")
            self.stdout.write("  Classification Report:")
            report = classification_report(y_test, y_pred, target_names=["Non-diabetic", "Diabetic"])
            for line in report.strip().split("\n"):
                self.stdout.write(f"    {line}")

            cm = confusion_matrix(y_test, y_pred)
            self.stdout.write(f"  Confusion Matrix:")
            self.stdout.write(f"    True Neg  {cm[0][0]:3d} | False Pos {cm[0][1]:3d}")
            self.stdout.write(f"    False Neg {cm[1][0]:3d} | True Pos  {cm[1][1]:3d}")

        # ── 8. Feature importances (Random Forest only) ───────────────────────────
        rf_model = results["Random Forest"]["pipeline"].named_steps["model"]
        importances = rf_model.feature_importances_
        self.stdout.write("\n── Random Forest Feature Importances ──")
        sorted_feats = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
        for feat, imp in sorted_feats:
            bar = "█" * int(imp * 40)
            self.stdout.write(f"  {feat:<22} {imp:.3f}  {bar}")

        # ── 9. Save the better model ──────────────────────────────────────────────
        best_name = max(results, key=lambda n: results[n]["auc"])
        best_pipeline = results[best_name]["pipeline"]
        best_auc = results[best_name]["auc"]

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({
            "pipeline": best_pipeline,
            "feature_cols": FEATURE_COLS,
            "model_name": best_name,
            "auc": best_auc,
        }, MODEL_PATH)

        self.stdout.write(
            self.style.SUCCESS(
                f"\nSaved '{best_name}' (AUC={best_auc:.3f}) to:\n  {MODEL_PATH}"
            )
        )
