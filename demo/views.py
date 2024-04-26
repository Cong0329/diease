from django.shortcuts import render
from .models import *
import csv

# Create your views here.

def home(request):
    # data = []
    # with open('data/Disease_symptom_and_patient_profile_dataset.csv', 'r') as file:
    #     reader = csv.DictReader(file)
    #     for row in reader:
    #         data.append(row)
    
    # context = {'data': data}
    return render(request, 'app/home.html')

# def process_csv(request):
#     with open('static/data/Disease_symptom_and_patient_profile_dataset.csv', 'r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             patient = PatientData(
#                 disease=row['Disease'],
#                 fever=row['Fever'],
#                 cough=row['Cough'],
#                 fatigue=row['Fatigue'],
#                 difficulty_breathing=row['Difficulty Breathing'],
#                 age=row['Age'],
#                 gender=row['Gender'],
#                 blood_pressure=row['Blood Pressure'],
#                 cholesterol_level=row['Cholesterol Level'],
#                 outcome_variable=row['Outcome Variable']
#             )
#             patient.save()
#     return render(request, 'app/home.html')
