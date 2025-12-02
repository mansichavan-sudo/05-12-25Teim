from django.db import models

class SentMessageLog(models.Model):
    channel = models.CharField(max_length=20)  # whatsapp / email
    customer_name = models.CharField(max_length=255)
    customer_phone = models.CharField(max_length=20, null=True, blank=True)
    customer_email = models.CharField(max_length=255, null=True, blank=True)
    message = models.TextField()
    status = models.CharField(max_length=20)  # success / failed
    provider = models.CharField(max_length=20)  # rapbooster / smtp
    message_id = models.CharField(max_length=100, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
