from django import forms
from .models import UploadFile, UploadImage, DropDownListPreprocessing, DropDownListGenerative
from django.db import models

class UploadFileForm(forms.ModelForm):

    class Meta:
        model = UploadFile
        fields = ("file","field")

class UploadImageForm(forms.ModelForm):

    class Meta:

        model = UploadImage
        fields = "__all__"


class CreateDropMenuForPreprocessing(forms.ModelForm):

    class Meta:

        model = DropDownListPreprocessing
        fields = "__all__"
        


class CreateDropMenuForGenerative(forms.ModelForm):

    class Meta:

        model = DropDownListGenerative
        fields = "__all__"







