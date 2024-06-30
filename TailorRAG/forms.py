from django import forms
from .models import Texte

class TexteForm(forms.ModelForm):
    class Meta:
        model = Texte
        fields = ['title', 'text']
