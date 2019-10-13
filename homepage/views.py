from django.shortcuts import render, render_to_response
from django.http import HttpResponse, JsonResponse
from django.template import RequestContext
from matplotlib.backends.backend_agg import FigureCanvasAgg
from django.core.files.storage import FileSystemStorage
import os, io
import pandas as pd
from .forms import UploadFileForm, UploadImageForm, CreateDropMenuForPreprocessing, CreateDropMenuForGenerative
import matplotlib.pyplot as plt
from .Pipelines import preprocessing_pipeline
from random import randint
from django.views.generic import TemplateView
from chartjs.views.lines import BaseLineChartView
#from backend.vae import main as variational_autoencoder
# Create your views here.

def home(request):

    return render(request, "index.html")

def link1(request):

    return render(request, "link1.html")

def contact(request):

    return render(request, "contact.html")

def apps(request):

    return render(request, "apps.html")

def blogs(request):

    return render(request, "blogs.html")

def about_me(request):

    return render(request, "about_me.html")

def tutorials(request):

    return render(request, "tutorials.html")

def quantum_classifier_blog(request):

    return render(request, "blogs/quantum_classifier.html")

def vae(request):
    return render(request, "variational_autoencoder_tensorflow.html")

def preprocessing(request):
    print("methd preproc ", request.method)
    form = CreateDropMenuForPreprocessing()
    if request.method == 'POST':
        form = CreateDropMenuForPreprocessing(request.POST)
        if form.is_valid():
            data= form.cleaned_data.get("IMPUTER")

        uploaded_file = request.FILES["myfile"]
        fs = FileSystemStorage()
        csv_name = fs.save(uploaded_file.name, uploaded_file)
        csv_path = fs.path(csv_name)
        df = pd.read_csv(csv_path, sep=",")
        media_path = "/Users/rubencr/Desktop/website/website/media"
        preprocessed_df = preprocessing_pipeline(df, path=media_path)
        preprocessed_df.to_csv(csv_path.split(".")[0]+"_preprocessed.csv")
        print(preprocessed_df)

    return render(request, "apps/preprocessing.html", {"form_p": form})

def generative(request):

    forms = CreateDropMenuForGenerative()
    if request.method == 'POST':
        forms = CreateDropMenuForGenerative(request.POST)
    elif request.method == 'GET':
        print(request.GET.get('generative'))
        return render(request, "apps/generative.html", {"form_g": forms})
    else:
        forms = CreateDropMenuForGenerative()
        print("none")
    

def upload_image(request, plot):

    buffer = io.BytesIO()
    canvas = FigureCanvasAgg(plot)
    buffer.getvalue()
    plot.savefig(buffer, format='png')
    plot.clear()

    return plot

def post_image(request):

    form = UploadImageForm()
    if request.POST == "POST":
        form = UploadImageForm(request.POST)
    
    return render(request, "apps.html", {"form_p": form})
    
class LineChartJSONView(BaseLineChartView):

    def __init__(self):

        self.path = "/Users/rubencr/Desktop/new.csv"
        self.df = pd.read_csv(self.path)

    def get_labels(self):
        """Return 7 labels for the x-axis."""
        return ["" for _ in range(len(self.df["ligand"].values))]

    def get_providers(self):
        """Return names of datasets."""
        return ["sasa","distance","energy"]

    def get_data(self):
        """Return 3 datasets to plot."""
        return [list(self.df["sasa"].values), list(self.df["distance"].values), list(self.df["energy"].values)]

        #return [self.df["sasa"].values, self.df["distance"].values, self.df["energy"].values]

line_chart = TemplateView.as_view(template_name='line_chart.html')
line_chart_json = LineChartJSONView.as_view()