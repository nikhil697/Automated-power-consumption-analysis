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