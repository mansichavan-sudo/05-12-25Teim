# crmapp/apps.py
from django.apps import AppConfig

class CrmappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'crmapp'

    def ready(self):
        """
        Import signals safely when the app is ready.
        """
        try:
            import crmapp.signals
        except Exception as e:
            import logging
            logging.warning(f"⚠️ Could not import crmapp.signals: {e}")
