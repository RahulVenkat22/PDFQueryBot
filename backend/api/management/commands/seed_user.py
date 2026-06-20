import os

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

User = get_user_model()


class Command(BaseCommand):
    """Seed a default admin login if no users exist yet.

    Credentials are read from the environment (with sensible defaults):
      SEED_ADMIN_USERNAME (default: admin)
      SEED_ADMIN_PASSWORD (default: Admin@12345)
      SEED_ADMIN_EMAIL    (default: admin@pdfquerybot.local)

    Run with ``--force`` to (re)set the password even if the user exists.
    """

    help = "Create a default admin user for logging in (idempotent)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Reset the password even if the user already exists.",
        )

    def handle(self, *args, **options):
        username = os.getenv("SEED_ADMIN_USERNAME", "admin")
        password = os.getenv("SEED_ADMIN_PASSWORD", "Admin@12345")
        email = os.getenv("SEED_ADMIN_EMAIL", "admin@pdfquerybot.local")

        # Only seed when the table is empty, unless --force is passed.
        if User.objects.exists() and not options["force"]:
            self.stdout.write(
                self.style.WARNING("Users already exist — skipping seed (use --force to reset).")
            )
            return

        user, created = User.objects.get_or_create(
            username=username,
            defaults={"email": email, "is_staff": True, "is_superuser": True},
        )
        user.email = email
        user.is_staff = True
        user.is_superuser = True
        user.set_password(password)
        user.save()

        action = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} admin user:"))
        self.stdout.write(f"  username: {username}")
        self.stdout.write(f"  password: {password}")
