from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.http import HttpResponse
from .forms import CustomUserCreationForm, TaskForm
from .models import Task, Profile

def index(request):
    return render(request , 'home.html')

def task_list(request):
    user_class = None
    tasks = None
    
    if request.user.is_authenticated:
        
        try:
            user_class = request.user.profile.user_class
            
            tasks = Task.objects.filter(user_class=user_class)
            
        except Profile.DoesNotExist:
            pass
    return render(request, 'task_list.html', {'user_class': user_class, 'tasks': tasks})

def add_task(request):
    if request.method == 'POST':
        form = TaskForm(request.POST)
        if form.is_valid():
            task = form.save(commit=False)
            task.created_by = request.user
            task.save()
            return redirect('task_list')  # Redirect to the task list page after successful task creation
    else:
        form = TaskForm()
    return render(request, 'add_task.html', {'form': form})

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')  # Redirect to the home page after successful registration
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('task_list')  # Redirect to the task list page after successful login
    else:
        form = AuthenticationForm()
    return render(request, 'registration/login.html', {'form': form})

def user_logout(request):
    logout(request)
    return redirect('index')
