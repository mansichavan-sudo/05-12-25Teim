from django.urls import path
from crmapp.views_message_templates import (
    messaging_page,
    get_template_content,
    send_whatsapp_message_api,
    send_email_message_api,
) 

urlpatterns = [
    path("", messaging_page, name="messaging_page"),

    path("api/get-template/<int:template_id>/<int:customer_id>/", 
         get_template_content),

    path("api/send/whatsapp/<int:customer_id>/",
         send_whatsapp_message_api),

    path("api/send/email/<int:customer_id>/",
         send_email_message_api),
]
