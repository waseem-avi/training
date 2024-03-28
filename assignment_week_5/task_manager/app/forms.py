# app/forms.py

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Task, Profile

class CustomUserCreationForm(UserCreationForm):
    class_choices = (
        ('java', 'Java'),
        ('python', 'Python'),
        ('mern', 'MERN'),
    )
    user_class = forms.ChoiceField(choices=class_choices)

    class Meta:
        model = User
        fields = ('username', 'password1', 'password2', 'user_class')
    def save(self, commit=True):
        user = super(CustomUserCreationForm, self).save(commit=False)
        user.save()
        profile = Profile.objects.create(user=user, user_class=self.cleaned_data['user_class'])
        return user

class TaskForm(forms.ModelForm):
    class Meta:
        model = Task
        fields = ['title', 'description', 'user_class']

    
