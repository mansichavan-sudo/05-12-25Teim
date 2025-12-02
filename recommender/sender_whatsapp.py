import requests
from notifications.models import SentMessageLog

API_KEY = "6538c8eff027d41e9151"
RAPBOOSTER_SEND_URL = "https://rapbooster.in/api/v1/sendMessage"

def send_whatsapp_message(customer, message):
    payload = {
        "apikey": API_KEY,
        "phone": customer.mobile_no,
        "message": message
    }

    try:
        response = requests.post(RAPBOOSTER_SEND_URL, json=payload)
        data = response.json()

        status = "success" if data.get("status") == "success" else "failed"

        SentMessageLog.objects.create(
            channel="whatsapp",
            customer_name=customer.customer_name,
            customer_phone=customer.mobile_no,
            message=message,
            status=status,
            provider="rapbooster",
            message_id=data.get("message_id", "")
        )

        return data

    except Exception as e:
        SentMessageLog.objects.create(
            channel="whatsapp",
            customer_name=customer.customer_name,
            customer_phone=customer.mobile_no,
            message=message,
            status="failed",
            provider="rapbooster"
        )
        return {"status": "failed", "error": str(e)}
