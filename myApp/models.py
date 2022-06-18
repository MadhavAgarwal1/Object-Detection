from django.db import models

# Create your models here.
class Detector(models.Model):
    image = models.ImageField(upload_to='myApp/images')