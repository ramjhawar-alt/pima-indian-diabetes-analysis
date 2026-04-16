"""
Django Admin registration for the Patient model.

The admin site is one of Django's most powerful features: by registering a
model here, Django auto-generates a full CRUD web interface for it — no HTML
or forms needed. You can browse, search, filter, and edit all 768 patients at
http://127.0.0.1:8000/admin/

ModelAdmin lets you customize how the model is displayed in the admin UI.
"""

from django.contrib import admin

from .models import Patient


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    # list_display: columns shown in the "all patients" table
    list_display = (
        "id",
        "age",
        "glucose",
        "bmi",
        "blood_pressure",
        "pregnancies",
        "diabetes_pedigree",
        "outcome_display",
    )

    # list_filter: adds a sidebar with clickable filters
    list_filter = ("outcome",)

    # search_fields: enables the search box (searches these columns via SQL LIKE)
    search_fields = ("age",)

    # ordering: default sort order in the list view
    ordering = ("id",)

    # list_per_page: how many rows per page
    list_per_page = 50

    # readonly_fields: shown on the detail page but cannot be edited
    readonly_fields = ("id",)

    def outcome_display(self, obj):
        """Custom column that shows a readable label instead of True/False."""
        return "Diabetic" if obj.outcome else "Non-diabetic"

    outcome_display.short_description = "Outcome"
    outcome_display.admin_order_field = "outcome"
