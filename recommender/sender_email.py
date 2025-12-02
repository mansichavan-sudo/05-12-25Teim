from django.core.mail import send_mail
from notifications.models import SentMessageLog
from django.conf import settings

def send_recommendation_email(customer, message):
    subject = "Your Pest Control Product Recommendations"
    recipient = customer.email

    try:
        send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [recipient])

        SentMessageLog.objects.create(
            channel="email",
            customer_name=customer.customer_name,
            customer_email=recipient,
            message=message,
            status="success",
            provider="smtp"
        )

        return {"status": "success"}

    except Exception as e:
        SentMessageLog.objects.create(
            channel="email",
            customer_name=customer.customer_name,
            customer_email=recipient,
            message=message,
            status="failed",
            provider="smtp"
        )

        return {"status": "failed", "error": str(e)}
