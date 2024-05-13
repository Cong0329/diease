from django.db import models

# Create your models here.

class PatientData(models.Model):
    name = models.CharField(max_length = 50, null=True)
    age = models.PositiveIntegerField()
    disease = models.CharField(max_length=100)
    fever = models.CharField(max_length=10)
    cough = models.CharField(max_length=10)
    fatigue = models.CharField(max_length=10)
    difficulty_breathing = models.CharField(max_length=10)
    gender = models.CharField(max_length=10) 
    blood_pressure = models.CharField(max_length=20)  
    cholesterol_level = models.CharField(max_length=20)
    predicted_result = models.CharField(max_length=10, blank=True, null=True)

    def __str__(self):
        return f"Patient {self.id}: {self.disease}"
    
