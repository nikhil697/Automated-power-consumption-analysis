# Generated by Django 5.0.7 on 2024-09-18 11:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0004_delete_electric'),
    ]

    operations = [
        migrations.CreateModel(
            name='Reading',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Timestamps', models.DateTimeField()),
                ('ampere', models.DecimalField(decimal_places=1, max_digits=6)),
                ('wattage_kwh', models.DecimalField(decimal_places=2, max_digits=6)),
                ('pf', models.DecimalField(decimal_places=3, max_digits=5)),
            ],
            options={
                'db_table': 'readings',
            },
        ),
    ]
