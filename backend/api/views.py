from rest_framework import filters, viewsets

from .models import User
from .serializers import UserSerializer


class UserViewSet(viewsets.ModelViewSet):
    """Full CRUD API for users.

    Provides ``list``, ``create``, ``retrieve``, ``update``, ``partial_update``
    and ``destroy`` actions out of the box.
    """

    queryset = User.objects.all()
    serializer_class = UserSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["first_name", "last_name", "mail_id", "phone_number"]
    ordering_fields = ["created_at", "first_name", "last_name"]
    ordering = ["-created_at"]
