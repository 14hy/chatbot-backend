from django.db import models

# Create your models here.


class Member(models.Model):
    name = models.CharField(max_length=200)
    mail = models.CharField(max_length=200)
    age = models.IntegerField(default=0)