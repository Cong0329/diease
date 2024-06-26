# Generated by Django 5.0 on 2024-04-26 06:28

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PatientData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('disease', models.CharField(max_length=100)),
                ('fever', models.BooleanField(default=False)),
                ('cough', models.BooleanField(default=False)),
                ('fatigue', models.BooleanField(default=False)),
                ('difficulty_breathing', models.BooleanField(default=False)),
                ('age', models.PositiveIntegerField()),
                ('gender', models.CharField(max_length=10)),
                ('blood_pressure', models.CharField(max_length=20)),
                ('cholesterol_level', models.CharField(max_length=20)),
                ('outcome_variable', models.CharField(max_length=100)),
            ],
        ),
    ]
