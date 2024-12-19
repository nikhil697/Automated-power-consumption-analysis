from django.db import models
import os
from django.core.validators import MinLengthValidator

# Create your models here.
class credent(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=20,primary_key=True)
    email_address = models.EmailField(unique=True)
    password = models.CharField(max_length=100, validators=[MinLengthValidator(8)])
    class Meta:
        db_table = 'credentials'

# class Electric(models.Model):
#     DATE = models.DateField()
#     TR_NO = models.IntegerField()
#     TIME = models.CharField(max_length=5)
#     VOLTAGE = models.DecimalField(max_digits=6, decimal_places=2)
#     AMP = models.DecimalField(max_digits=6, decimal_places=2)
#     KWH = models.DecimalField(max_digits=10, decimal_places=2)
#     O_L = models.DecimalField(max_digits=4, decimal_places=2)
#     Sixth_FLOOR_CS = models.DecimalField(max_digits=6, decimal_places=2)
#     PROFESSOR_QUATERS = models.DecimalField(max_digits=6, decimal_places=2)
#     HOSTEL_N_ROAD_SIDE = models.DecimalField(max_digits=6, decimal_places=2)
#     lc_1 = models.DecimalField(max_digits=6, decimal_places=2)
#     lc_2 = models.DecimalField(max_digits=6, decimal_places=2)
#     class Meta:
#         db_table = 'elecread'

class Reading(models.Model):
    Timestamps = models.DateTimeField()
    ampere = models.DecimalField(max_digits=6, decimal_places=1)
    wattage_kwh = models.DecimalField(max_digits=6, decimal_places=2)
    pf = models.DecimalField(max_digits=5, decimal_places=3)
    class Meta:
        db_table = 'readings'
