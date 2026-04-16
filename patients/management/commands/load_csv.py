"""
Custom Django management command: python manage.py load_csv

This is how Django lets you write one-off scripts that have full access to the
ORM (database), settings, and all your models — without running a web server.

Run with:
    python manage.py load_csv
    python manage.py load_csv --path /some/other/path/diabetes.csv
    python manage.py load_csv --clear   # wipe table before loading
"""

import csv
import os

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from patients.models import Patient

# Column order in the CSV file (no header row exists in the raw dataset)
CSV_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
    "outcome",
]


class Command(BaseCommand):
    """
    BaseCommand is the Django class every management command must inherit from.
    self.stdout.write() prints to the terminal (not print()) so Django can
    capture/suppress output during tests.
    """

    help = "Load the Pima Indian Diabetes CSV into the patients_patient table"

    def add_arguments(self, parser):
        """
        add_arguments lets you define CLI flags for your command.
        'parser' works exactly like Python's argparse.ArgumentParser.
        """
        parser.add_argument(
            "--path",
            default=os.path.join(settings.BASE_DIR, "diabetes.csv"),
            help="Path to the CSV file (default: diabetes.csv in project root)",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete all existing patients before loading",
        )

    def handle(self, *args, **options):
        """
        handle() is the entry point — Django calls this when you run the command.
        """
        csv_path = options["path"]

        if not os.path.exists(csv_path):
            raise CommandError(
                f"CSV file not found at: {csv_path}\n"
                "Place diabetes.csv in the project root or pass --path <file>"
            )

        if options["clear"]:
            count, _ = Patient.objects.all().delete()
            self.stdout.write(self.style.WARNING(f"Deleted {count} existing patients."))

        self.stdout.write(f"Reading {csv_path} …")

        patients_to_create = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

            for line_num, row in enumerate(reader, start=1):
                if len(row) != 9:
                    self.stdout.write(
                        self.style.WARNING(f"  Skipping line {line_num}: expected 9 columns, got {len(row)}")
                    )
                    continue

                # Map CSV columns to Patient fields
                patients_to_create.append(
                    Patient(
                        pregnancies=int(row[0]),
                        glucose=float(row[1]),
                        blood_pressure=float(row[2]),
                        skin_thickness=float(row[3]),
                        insulin=float(row[4]),
                        bmi=float(row[5]),
                        diabetes_pedigree=float(row[6]),
                        age=int(row[7]),
                        outcome=bool(int(row[8])),
                    )
                )

        # bulk_create sends a single INSERT statement with all rows — much faster
        # than calling Patient.objects.create() 768 times in a loop.
        created = Patient.objects.bulk_create(patients_to_create)

        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully loaded {len(created)} patients into the database."
            )
        )
