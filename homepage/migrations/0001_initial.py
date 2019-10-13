# -*- coding: utf-8 -*-
# Generated by Django 1.11.24 on 2019-10-13 20:47
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DropDownListGenerative',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('generative', models.CharField(choices=[(b'AUTO', b'Variational Autoencoder'), (b'GAN', b'Generative Adversarial Networks'), (b'Lstm', b'LSTM')], default=b'AUTO', max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='DropDownListPreprocessing',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('split', models.CharField(choices=[(b'YES', b'Yes'), (b'NO', b'No')], default=b'YES', max_length=10)),
                ('imputer', models.CharField(choices=[(b'YES', b'Yes'), (b'NO', b'No')], default=b'FREQUENT', max_length=10)),
                ('outlier', models.CharField(choices=[(b'NONE', b'None'), (b'STANDARD', b'Standard'), (b'INTERQUANTILE', b'Interquantile'), (b'ELLIPTIC', b'Elliptic')], default=b'NONE', max_length=10)),
            ],
        ),
        migrations.CreateModel(
            name='DropDownMenu',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='UploadFile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('field', models.FileField(upload_to=b'')),
                ('file', models.CharField(max_length=10)),
            ],
        ),
        migrations.CreateModel(
            name='UploadImage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cover', models.ImageField(upload_to=b'')),
            ],
        ),
    ]
