from django.conf import settings
from django.db import models
from django.utils import timezone


class UploadFile(models.Model):

    field = models.FileField(max_length=100)
    file = models.CharField(max_length=10)

class DropDownMenu(models.Model):

    text = models.TextField(max_length=50)

class UploadImage(models.Model):

    #title = models.TextField()
    cover = models.ImageField()
    
class DropDownListPreprocessing(models.Model):

    SCALERS = (
        ("ROBUST", "Robust"),
        ("STANDARD", "Standard"),
        ("MINMAX", "MinMax")
    )

    SPLIT = (
        ("YES", "Yes"),
        ("NO", "No")
    )

    IMPUTER = (
        ("FREQUENT", "Frequent"),
        ("MEDIAN", "Median"),
        ("MEAN", "Mean")
    )

    OUTLIERS = (

        ("NONE", "None"),
        ("STANDARD", "Standard"),
        ("INTERQUANTILE", "Interquantile"),
        ("ELLIPTIC", "Elliptic" )


    )

    #fiel = models.TextField()
    #scaler = models.CharField(max_length=6, choices=SCALERS, default="STANDARD")
    split = models.CharField(max_length=10, choices=SPLIT, default="YES")
    imputer = models.CharField(max_length=10, choices=SPLIT, default="FREQUENT")
    outlier =  models.CharField(max_length=10, choices=OUTLIERS, default="NONE")

class DropDownListGenerative(models.Model):

    GENERATIVE = (

        ("AUTO", "Variational Autoencoder"),
        ("GAN", "Generative Adversarial Networks"),
        ("Lstm", "LSTM")

    )

    generative = models.CharField(max_length=20, choices=GENERATIVE, default="AUTO")

    def __str__(self):
        return self.generative