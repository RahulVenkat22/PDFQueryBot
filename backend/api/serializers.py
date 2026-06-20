import re

from rest_framework import serializers

from .models import User

PHONE_RE = re.compile(r"^\+?[0-9]{7,15}$")


class UserSerializer(serializers.ModelSerializer):
    """Serializer for the :class:`~api.models.User` model."""

    full_name = serializers.CharField(read_only=True)

    class Meta:
        model = User
        fields = [
            "id",
            "first_name",
            "last_name",
            "full_name",
            "mail_id",
            "phone_number",
            "gender",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def validate_first_name(self, value):
        value = value.strip()
        if not value:
            raise serializers.ValidationError("First name cannot be blank.")
        return value

    def validate_last_name(self, value):
        return value.strip()

    def validate_mail_id(self, value):
        return value.strip().lower()

    def validate_phone_number(self, value):
        value = value.strip()
        if not PHONE_RE.match(value):
            raise serializers.ValidationError(
                "Enter a valid phone number (7-15 digits, optional leading +)."
            )
        return value
