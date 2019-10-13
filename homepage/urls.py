"""website URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
from django.conf.urls import include, url
from . import views

urlpatterns = [
    url(r'^$', views.home),
    url(r'^index.html$', views.home),
    url(r'link1.html$', views.link1),
    url(r'contact.html$', views.contact),
    url(r'line_chart/$', views.preprocessing),
    url(r'line_chart/$', views.line_chart),
    url(r'apps.html$', views.apps),
    url(r'blogs.html$', views.blogs),
    url(r'apps/preprocessing.html$', views.preprocessing),
    url(r'^apps/generative.html$', views.generative),
    url(r'^blogs/quantum_classifier.html$', views.quantum_classifier_blog),
    url(r'^variational_autoencoder_tensorflow.html$', views.vae), 
    url(r'about_me.html$', views.about_me),
    url(r'tutorials.html$', views.tutorials),
    #url(r'apps/$', views.preprocessing),
    url(r'^line_chart/json/$', views.line_chart_json,
        name='line_chart_json')


]
