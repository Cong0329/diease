# Generated by Django 5.0 on 2024-05-13 15:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0007_delete_patientname'),
    ]

    operations = [
        migrations.AlterField(
            model_name='patientdata',
            name='cough',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='patientdata',
            name='difficulty_breathing',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='patientdata',
            name='fatigue',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='patientdata',
            name='fever',
            field=models.BooleanField(default=False),
        ),
    ]
