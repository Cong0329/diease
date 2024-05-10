from django.shortcuts import render
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


def disease_prediction(request):
    patients = PatientData.objects.all()
    context = {'patients': patients}
    return render(request, 'app/prediction.html', context)



def predict_result(request):
    if request.method == 'POST':
        # Load data
        df = pd.read_csv("data.csv")

        # Handling missing and duplicate values
        df.isnull().sum()
        df.duplicated().sum()
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        df = df[(df['Age'] >= 20) & (df['Age'] <= 70)]
        df.reset_index(drop=True, inplace=True)

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

        # New input for prediction
        new_input = {
            'Disease': int(request.POST.get('Disease', 0)),
            'Fever': int(request.POST.get('Fever', 0)),
            'Cough': int(request.POST.get('Cough', 0)),
            'Fatigue': int(request.POST.get('Fatigue', 0)),
            'Difficulty Breathing': int(request.POST.get('DifficultyBreathing', 0)),
            'Age': int(request.POST.get('Age', 0)),
            'Gender': int(request.POST.get('Gender', 0)),
            'Blood Pressure': int(request.POST.get('BloodPressure', 0)),
            'Cholesterol Level': int(request.POST.get('CholesterolLevel', 0)),
        }


        # Convert to DataFrame
        new_input_df = pd.DataFrame(new_input, index=[0])

        # Predict result for new input
        predicted_result = model.predict(new_input_df)

        # Render HTML template with predicted result
        if predicted_result == 1:
            result = "Positive"
        else:
            result = "Negative"

        html_content = f"<html><body><h1>Predicted Result: {result}</h1></body></html>"

        # Return HTML response
        return HttpResponse(html_content)

    else:
        return HttpResponse('Only POST requests are allowed.')


