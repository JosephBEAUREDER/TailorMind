from django.apps import AppConfig


class TailorragConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'TailorRAG'

    def ready(self):
        import TailorRAG.models