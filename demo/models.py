from django.db import models

# Create your models here.


class PatientData(models.Model):
    name = models.CharField(max_length = 50, null=True)
    age = models.PositiveIntegerField()
    disease = models.CharField(max_length=100)
    fever = models.BooleanField(default=False)
    cough = models.BooleanField(default=False)
    fatigue = models.BooleanField(default=False)
    difficulty_breathing = models.BooleanField(default=False)
    gender = models.CharField(max_length=10) 
    blood_pressure = models.CharField(max_length=20)  
    cholesterol_level = models.CharField(max_length=20)  

    def __str__(self):
        return f"Patient {self.id}: {self.disease}"
    
