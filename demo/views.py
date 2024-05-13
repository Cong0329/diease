from django.shortcuts import render, redirect
from sklearn.metrics import f1_score, make_scorer, mean_absolute_error, precision_score
from .models import *
from django.http import HttpResponse, JsonResponse
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
# # Create your views here.

def home(request):
    patients = PatientData.objects.all()
    context = {'patients': patients}
    return render(request, 'app/home.html', context)


def predict_result(request):
    if request.method == 'POST':
        # Load data
        df = pd.read_csv("data.csv")

        disease_names = df['Disease'].unique()
        fever_names = df['Fever'].unique()
        cough_names = df['Cough'].unique()
        fatigue_names = df['Fatigue'].unique()
        difficultybreathing_names = df['Difficulty Breathing'].unique()
        bloodpressure_names = df['Blood Pressure'].unique()
        cholesterollevel_names = df['Cholesterol Level'].unique()

        # Handling missing and duplicate values
        df.isnull().sum()
        df.duplicated().sum()
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        df = df[(df['Age'] >= 20) & (df['Age'] <= 70)]
        df.reset_index(drop=True, inplace=True)

        disease_mapping = {disease_name: i for i, disease_name in enumerate(disease_names)}
        fever_mapping = {fever_name: i for i, fever_name in enumerate(fever_names)}
        cough_mapping = {cough_name: i for i, cough_name in enumerate(cough_names)}
        fatigue_mapping = {fatigue_name: i for i, fatigue_name in enumerate(fatigue_names)}
        difficultybreathing_mapping = {difficultybreathing_name: i for i, difficultybreathing_name in enumerate(difficultybreathing_names)}
        bloodpressure_mapping = {bloodpressure_name: i for i, bloodpressure_name in enumerate(bloodpressure_names)}
        cholesterollevel_mapping = {cholesterollevel_name: i for i, cholesterollevel_name in enumerate(cholesterollevel_names)}


        # Encoding categorical features
        x_encoder = OneHotEncoder(sparse_output=False, drop='first')
        y_encoder = LabelEncoder()

        df['Fever'] = x_encoder.fit_transform(df[['Fever']])
        df['Cough'] = x_encoder.fit_transform(df[['Cough']])
        df['Fatigue'] = x_encoder.fit_transform(df[['Fatigue']])
        df['Difficulty Breathing'] = x_encoder.fit_transform(df[['Difficulty Breathing']])
        df['Gender'] = x_encoder.fit_transform(df[['Gender']])
        df['Blood Pressure'] = x_encoder.fit_transform(df[['Blood Pressure']])
        df['Cholesterol Level'] = x_encoder.fit_transform(df[['Cholesterol Level']])

        df['Disease'] = y_encoder.fit_transform(df[['Disease']])
        df['Outcome Variable'] = y_encoder.fit_transform(df[['Outcome Variable']])

        X = df.drop(['Outcome Variable'], axis=1)
        y = df['Outcome Variable']

        # Splitting data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Initialize Logistic Regression model
        LR = LogisticRegression()

        # Train the model
        model = LR.fit(X_train, y_train)

        predictor_columns = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
                     'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']


        # New input for prediction
        new_input = {
            'Name': request.POST.get('Name', ''),
            'Disease': request.POST.get('Disease', ''),
            'Fever': request.POST.get('Fever', 0),
            'Cough': request.POST.get('Cough', 0),
            'Fatigue': request.POST.get('Fatigue', 0),
            'Difficulty Breathing': request.POST.get('DifficultyBreathing', 0),
            'Age': int(request.POST.get('Age', 0)),
            'Gender': int(request.POST.get('Gender', 0)),
            'Blood Pressure': request.POST.get('BloodPressure', 0),
            'Cholesterol Level': request.POST.get('CholesterolLevel', 0),
        }
        
        disease_encoded = disease_mapping.get(new_input['Disease'], -1)
        new_input['Disease'] = disease_encoded

        fever_encoded = fever_mapping.get(new_input['Fever'], -1)
        new_input['Fever'] = fever_encoded

        cough_encoded = cough_mapping.get(new_input['Cough'], -1)
        new_input['Cough'] = cough_encoded

        fatigue_encoded = fatigue_mapping.get(new_input['Fatigue'], -1)
        new_input['Fatigue'] = fatigue_encoded

        difficultybreathing_encoded = difficultybreathing_mapping.get(new_input['Difficulty Breathing'], -1)
        new_input['Difficulty Breathing'] = difficultybreathing_encoded

        bloodpressure_encoded = bloodpressure_mapping.get(new_input['Blood Pressure'], -1)
        new_input['Blood Pressure'] = bloodpressure_encoded

        cholesterollevel_encoded = cholesterollevel_mapping.get(new_input['Cholesterol Level'], -1)
        new_input['Cholesterol Level'] = cholesterollevel_encoded

        print("New Input:", new_input)
        
        # Convert to DataFrame
        new_input_df = pd.DataFrame(new_input, index=[0])[predictor_columns]

        # Predict result for new input
        predicted_result = model.predict(new_input_df)

        # Save data to database
        patient_data = PatientData(
            name=request.POST.get('Name'),
            disease=request.POST.get('Disease'),
            fever=request.POST.get('Fever'),
            cough=request.POST.get('Cough'),
            fatigue=request.POST.get('Fatigue'),
            difficulty_breathing=request.POST.get('DifficultyBreathing'),
            age=request.POST.get('Age'),
            gender=request.POST.get('Gender'),
            blood_pressure=request.POST.get('BloodPressure'),
            cholesterol_level=request.POST.get('CholesterolLevel'),
            predicted_result="Positive" if predicted_result == 1 else "Negative"
        )
        
        patient_data.save()

        # Render HTML template with predicted result
        html_content = f"<html><body><h1>Predicted Result: {'Positive' if predicted_result == 1 else 'Negative'}</h1></body></html>"

        # Return HTML response
        return HttpResponse(html_content)

    else:
        return HttpResponse('Only POST requests are allowed.')