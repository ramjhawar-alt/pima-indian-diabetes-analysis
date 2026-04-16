"""
Project-level URL configuration.

This is the root router. Think of it as a receptionist:
  - Requests starting with "admin/"  → go to Django's built-in admin
  - Requests starting with "patients/" → go to the patients app's urls.py
  - "stats/" → goes directly to the stats view
  - The root "/" → redirects to /patients/ for convenience

include() delegates everything after the prefix to another urls.py file,
keeping each app's routing self-contained.
"""

from django.contrib import admin
from django.shortcuts import redirect
from django.urls import include, path

from patients import views as patient_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("patients/", include("patients.urls", namespace="patients")),
    path("stats/", patient_views.stats, name="stats"),
    path("ask/", patient_views.ask, name="ask"),
    path("charts/", patient_views.charts, name="charts"),
    path("charts/api/", patient_views.charts_api, name="charts_api"),
    path("predict/", patient_views.predict, name="predict"),
    path("", lambda request: redirect("patients:list")),
]
