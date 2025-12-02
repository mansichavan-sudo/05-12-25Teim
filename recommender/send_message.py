import requests
from django.conf import settings
from crmapp.models import customer_details, MessageTemplates , SentMessageLog
from recommender.models import PestRecommendation


RAPBOOSTER_API_KEY = "6538c8eff027d41e9151"
RAPBOOSTER_URL = "https://backend.rapbooster.in/api/v1/send/message"


def send_recommendation_message(template_id, customer_id):
    """
    Send WhatsApp message using RapBooster API and auto-fill:
    - customer name
    - base product
    - recommended product
    Then save the message in SentMessageLog.
    """

    # --------------------------
    # 1. FETCH TEMPLATE
    # --------------------------
    template = MessageTemplates.objects.get(id=template_id)

    # --------------------------
    # 2. FETCH CUSTOMER
    # --------------------------
    customer = customer_details.objects.get(id=customer_id)

    # --------------------------
    # 3. FETCH RECOMMENDATION
    # --------------------------
    rec = PestRecommendation.objects.filter(customer_id=customer_id).first()

    base_product = rec.base_product.name if rec else ""
    recommended_product = rec.recommended_product.name if rec else ""

    # --------------------------
    # 4. AUTO-FILL TEMPLATE VARIABLES
    # --------------------------
    body = (
        template.body
        .replace("{{ customer_name }}", customer.customer_name)
        .replace("{{ product }}", base_product)
        .replace("{{ recommended_product }}", recommended_product)
    )

    subject = template.subject if template.subject else ""

    # --------------------------
    # 5. RAPBOOSTER PAYLOAD
    # --------------------------
    payload = {
        "message": body,
        "phone_numbers": [customer.phone],
        "type": "whatsapp"
    }

    headers = {
        "Authorization": RAPBOOSTER_API_KEY,
        "Content-Type": "application/json",
    }

    # --------------------------
    # 6. SEND MESSAGE
    # --------------------------
    try:
        response = requests.post(
            RAPBOOSTER_URL,
            json=payload,
            headers=headers,
            timeout=15
        )

        status = "sent" if response.status_code == 200 else "failed"

    except Exception as e:
        response = None
        status = "error"

    # --------------------------
    # 7. SAVE LOG PROPERLY
    # --------------------------
    SentMessageLog.objects.create(
        template=template,
        customer=customer,
        recipient=customer.phone,
        channel="whatsapp",
        rendered_subject=subject,
        rendered_body=body,
        status=status,
        provider_response=response.text if response else str(e),
    )

    return {
        "status": status,
        "customer": customer.customer_name,
        "sent_to": customer.phone,
        "message": body,
        "response": response.text if response else str(e),
    }
