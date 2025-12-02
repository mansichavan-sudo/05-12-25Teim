import requests
from django.core.mail import send_mail
from django.conf import settings

from crmapp.models import SentMessageLog

def create_log(customer, recipient, channel, subject, body, status, provider_response, message_id=None):
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


# ---------------------------------------------------------------
# WHATSAPP (RapBooster)
# ---------------------------------------------------------------
def send_whatsapp(customer, message):
    phone = str(customer.primarycontact)

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
        "apikey": settings.RAPBOOSTER_API_KEY,
        "phone": phone,
        "message": message
    }

    try:
        response = requests.post(settings.RAPBOOSTER_API_URL, json=payload, timeout=10)
        resp_json = response.json() if "json" in response.headers.get("content-type", "") else {}
        message_id = resp_json.get("message_id") or resp_json.get("id")

        log.status = "sent" if resp_json.get("status") == "success" else "failed"
        log.provider_response = response.text
        log.message_id = message_id
        log.save()

        return log.status, response.text

    except Exception as e:
        log.status = "error"
        log.provider_response = str(e)
        log.save()
        return "error", str(e)


# ---------------------------------------------------------------
# EMAIL
# ---------------------------------------------------------------
def send_email(customer, subject, message):
    recipient = customer.primaryemail

    log = create_log(
        customer=customer,
        recipient=recipient,
        channel="email",
        subject=subject,
        body=message,
        status="queued",
        provider_response=""
    )

    try:
        send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [recipient])
        log.status = "sent"
        log.provider_response = "Email sent"
        log.save()
        return "sent", "Email sent"

    except Exception as e:
        log.status = "error"
        log.provider_response = str(e)
        log.save()
        return "error", str(e)
