# Generated by Django 5.0 on 2024-05-13 15:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0008_alter_patientdata_cough_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='patientdata',
            name='cough',
            field=models.CharField(max_length=10),
        ),
        migrations.AlterField(
            model_name='patientdata',
            name='difficulty_breathing',
            field=models.CharField(max_length=10),
        ),
        migrations.AlterField(
            model_name='patientdata',
            name='fatigue',
            field=models.CharField(max_length=10),
        ),
        migrations.AlterField(
            model_name='patientdata',
            name='fever',
            field=models.CharField(max_length=10),
        ),
    ]
