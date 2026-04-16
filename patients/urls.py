"""
URL configuration for the patients app.

URL patterns are checked top-to-bottom. Django uses the first one that matches.

Each path() call maps a URL pattern to a view function and gives it a name.
The name is used in templates with {% url 'name' %} so you never hardcode URLs.
"""

from django.urls import path

from . import views

app_name = "patients"  # namespace — keeps URL names scoped to this app

urlpatterns = [
    path("", views.patient_list, name="list"),
    path("<int:pk>/", views.patient_detail, name="detail"),
]
