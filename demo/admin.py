from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from .models import *

# Register your models here.

# class Patient(ImportExportModelAdmin):
#     list_display = ('disease', 'fever', 'cough', 'fatigue', 'difficulty_breathing')