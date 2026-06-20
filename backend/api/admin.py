from django.contrib import admin

from .models import User


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ("id", "first_name", "last_name", "mail_id", "phone_number", "gender", "created_at")
    list_filter = ("gender", "created_at")
    search_fields = ("first_name", "last_name", "mail_id", "phone_number")
    ordering = ("-created_at",)
    readonly_fields = ("created_at", "updated_at")
