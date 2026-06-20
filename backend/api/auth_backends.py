from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend

User = get_user_model()


class EmailBackend(ModelBackend):
    """Authenticate by email address (case-insensitive) instead of username."""

    def authenticate(self, request, email=None, password=None, **kwargs):
        # Allow callers that still pass the value as ``username``.
        if email is None:
            email = kwargs.get("username")
        if email is None or password is None:
            return None

        try:
            user = User.objects.get(email__iexact=email)
        except User.DoesNotExist:
            # Run the hasher once to keep timing consistent (avoid user enumeration).
            User().set_password(password)
            return None
        except User.MultipleObjectsReturned:
            return None

        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None
