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
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^', include('homepage.urls')),
    url(r'^index.html', include('homepage.urls')),
    url(r'link1.html', include('homepage.urls')),
    url(r'contact.html', include('homepage.urls')),
    url(r'about_me.html', include('homepage.urls')),
    url(r'preprocessing.html', include('homepage.urls')),
    url(r'accounts/', include('django.contrib.auth.urls')),
    url(r'^line_chart/', include('homepage.urls')),
    url(r'^apps.html', include('homepage.urls')),
    url(r'blogs.html', include('homepage.urls')),
    url(r'tutorials.html', include('homepage.urls')),
    url(r'^line_chart/json/', include("homepage.urls")),
    url(r'apps/preprocessing.html', include("homepage.urls")),
    url(r'^apps/generative.html', include("homepage.urls")),
    url(r'^variational_autoencoder_tensorflow.html', include("homepage.urls")),	
    url(r'^blogs/quantum_classifier.html', include("homepage.urls"))
] +static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

