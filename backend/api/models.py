from django.db import models


class User(models.Model):
    """An application user managed through the CRUD API.

    This is a plain domain model (not Django's auth user). The ``id`` primary
    key is provided automatically by Django as a ``BigAutoField``.
    """

    class Gender(models.TextChoices):
        MALE = "male", "Male"
        FEMALE = "female", "Female"
        OTHER = "other", "Other"

    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    mail_id = models.EmailField("email address", unique=True)
    phone_number = models.CharField(max_length=20)
    gender = models.CharField(max_length=10, choices=Gender.choices)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["mail_id"]),
            models.Index(fields=["last_name", "first_name"]),
        ]

    def __str__(self) -> str:
        return f"{self.first_name} {self.last_name} <{self.mail_id}>"

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()
