from django.db import models

# Create your models here.
class Task(models.Model):
    
    room = models.ForeignKey(max_length=200)
    title = models.CharField(max_length=150)
    sub_date = models.DateTimeField()

    
    def __str__(self):
        return f"{self.title} assigned to {self.group}"