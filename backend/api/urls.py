from django.urls import path
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView

from .auth_views import EmailTokenObtainPairView, LogoutView, MeView
from .views import UserViewSet

router = DefaultRouter()
router.register(r"users", UserViewSet, basename="user")

auth_patterns = [
    path("auth/login/", EmailTokenObtainPairView.as_view(), name="login"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/logout/", LogoutView.as_view(), name="logout"),
    path("auth/me/", MeView.as_view(), name="me"),
]

urlpatterns = auth_patterns + router.urls
