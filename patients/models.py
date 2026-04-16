from django.db import models


class Patient(models.Model):
    """
    Represents one row from the Pima Indian Diabetes dataset.

    Each field maps directly to a column that Django will create in PostgreSQL.
    Django automatically adds an integer primary key called 'id'.
    """

    pregnancies = models.IntegerField(
        help_text="Number of times the patient has been pregnant"
    )
    glucose = models.FloatField(
        help_text="Plasma glucose concentration (mg/dL) from a 2-hour oral glucose tolerance test"
    )
    blood_pressure = models.FloatField(
        help_text="Diastolic blood pressure (mm Hg)"
    )
    skin_thickness = models.FloatField(
        help_text="Triceps skin fold thickness (mm)"
    )
    insulin = models.FloatField(
        help_text="2-hour serum insulin (μU/mL)"
    )
    bmi = models.FloatField(
        help_text="Body mass index (weight in kg / height in m²)"
    )
    diabetes_pedigree = models.FloatField(
        help_text="Diabetes pedigree function — a genetic risk score based on family history"
    )
    age = models.IntegerField(
        help_text="Age in years"
    )
    outcome = models.BooleanField(
        help_text="True = patient has diabetes, False = patient does not"
    )

    class Meta:
        ordering = ['id']
        verbose_name = "Patient"
        verbose_name_plural = "Patients"

    def __str__(self):
        status = "diabetic" if self.outcome else "non-diabetic"
        return f"Patient #{self.id} — age {self.age}, {status}"
