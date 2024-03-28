
from django.db import models
from django.contrib.auth.models import User

class Task(models.Model):
    CLASS_CHOICES = (
        ('java', 'Java'),
        ('python', 'Python'),
        ('mern', 'MERN'),
    )
    title = models.CharField(max_length=100)
    description = models.TextField()
    user_class = models.CharField(max_length=10, choices=CLASS_CHOICES)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    CLASS_CHOICES = (
        ('java', 'Java'),
        ('python', 'Python'),
        ('mern', 'MERN'),
    )
    user_class = models.CharField(max_length=10, choices=CLASS_CHOICES)

    def __str__(self):
        return self.user.username
