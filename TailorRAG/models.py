# TailorRAG/models.py
from django.db import models
from django.contrib.auth.models import User

class Texte(models.Model):
    title = models.CharField(max_length=255)
    text = models.TextField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.title
