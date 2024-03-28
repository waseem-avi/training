from django.contrib import admin
from .models import Task
from django.contrib.auth.models import User

@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    list_display = ['title', 'description', 'user_class', 'created_by']
    actions = ['add_task_to_users']

    def add_task_to_users(self, request, queryset):
        selected_tasks = queryset.values_list('title', flat=True)
        for task in queryset:
            users = User.objects.filter(profile__user_class=task.user_class)
            for user in users:
                Task.objects.create(
                    title=task.title,
                    description=task.description,
                    user_class=task.user_class,
                    created_by=user
                )
        self.message_user(request, f'Tasks {", ".join(selected_tasks)} added to users successfully.')

    add_task_to_users.short_description = "Add selected tasks to users"
