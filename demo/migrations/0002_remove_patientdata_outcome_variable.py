# Generated by Django 5.0 on 2024-05-05 09:51

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='patientdata',
            name='outcome_variable',
        ),
    ]
