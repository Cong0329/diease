# Generated by Django 5.0 on 2024-05-13 10:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0005_patientdata_predicted_result'),
    ]

    operations = [
        migrations.CreateModel(
            name='PatientName',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
            ],
        ),
    ]
