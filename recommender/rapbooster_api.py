import requests
import json
from django.core.mail import send_mail
from django.conf import settings

from crmapp.models import SentMessageLog, customer_details
import os


# ====================================================================
#   RAPBOOSTER PRODUCTION VARIABLES
# ====================================================================

API_URL = "https://api.rapbooster.com/v1/send"     # ✔ Correct Endpoint
API_KEY = os.getenv("RAPBOOSTER_API_KEY", "6538c8eff027d41e9151")   # ✔ Uses .env or fallback


# ====================================================================
#   UNIVERSAL MESSAGE LOGGER —  PERFECTLY MATCHES MODEL
# ====================================================================

def create_log(
    customer,
    recipient,
    channel,
    subject,
    body,
    status,
    provider_response,
    message_id=None
):
    """Writes into crmapp_sentmessagelog exactly matching DB structure."""

    return SentMessageLog.objects.create(
        customer=customer,
        recipient=recipient,
        channel=channel,
        rendered_subject=subject,
        rendered_body=body,
        status=status,
        provider_response=provider_response,
        message_id=message_id
    )


# ====================================================================
#                    RAPBOOSTER WHATSAPP API
# ====================================================================

def send_whatsapp_message(customer: customer_details, message: str):
    """
    Sends WhatsApp message using RapBooster API
    and logs everything.
    """

    phone = str(customer.primarycontact)

    # 1️⃣ Pre-log (queued)
    log = create_log(
        customer=customer,
        recipient=phone,
        channel="whatsapp",
        subject="",
        body=message,
        status="queued",
        provider_response=""
    )

    payload = {
        "apikey": API_KEY,
        "phone": phone,
        "message": message
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        resp_text = response.text

        try:
            resp_json = response.json()
        except:
            resp_json = {}

        message_id = resp_json.get("message_id") or resp_json.get("id")

        # 2️⃣ Update log after sending
        log.status = "sent" if (
            response.status_code == 200 and resp_json.get("status") == "success"
        ) else "failed"

        log.provider_response = resp_text
        log.message_id = message_id
        log.save()

        return log.status, resp_text

    except Exception as e:
        log.status = "error"
        log.provider_response = str(e)
        log.save()

        return "error", str(e)



# ====================================================================
#                           EMAIL SENDER
# ====================================================================

def send_email_message(customer: customer_details, subject: str, message: str):
    """
    Sends email via Django backend and logs it.
    """

    recipient_email = customer.primaryemail

    # 1️⃣ Pre-log
    log = create_log(
        customer=customer,
        recipient=recipient_email,
        channel="email",
        subject=subject,
        body=message,
        status="queued",
        provider_response=""
    )

    try:
        send_mail(
            subject,
            message,
            settings.DEFAULT_FROM_EMAIL,
            [recipient_email],
            fail_silently=False,
        )

        log.status = "sent"
        log.provider_response = "Email successfully sent."
        log.save()

        return "sent", "Email Sent"

    except Exception as e:
        log.status = "error"
        log.provider_response = str(e)
        log.save()

        return "error", str(e)



# ====================================================================
#                    SHORTCUT FOR RECOMMENDATION
# ====================================================================

def send_recommendation_message(customer: customer_details, message: str):
    return send_whatsapp_message(customer, message)
