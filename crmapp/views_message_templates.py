from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

from crmapp.models import MessageTemplates, customer_details
from recommender.models import PestRecommendation

from crmapp.utils.template_renderer import render_dynamic_template
from notifications.message_sender import send_whatsapp, send_email


# ============================================================
# 1️⃣ FETCH TEMPLATE + AUTO-FILL FIELDS
# ============================================================
def get_template_content(request, template_id, customer_id):
    try:
        # Fetch template
        template = MessageTemplates.objects.get(id=template_id)

        # Fetch customer
        customer = customer_details.objects.get(id=customer_id)

        # Fetch last generated recommendation
        rec = (
            PestRecommendation.objects
            .select_related("base_product", "recommended_product")
            .filter(customer_id=customer_id)
            .order_by("-created_at")
            .first()
        )

        base_product_name = rec.base_product.product_name if rec else ""
        recommended_product_name = rec.recommended_product.product_name if rec else ""

        # Prepare dynamic data
        data = {
            "template_id": template.id,
            "subject": template.subject or "",
            "body": template.body or "",
            "customer_name": customer.fullname,
            "product": base_product_name,
            "recommended_product": recommended_product_name,
        }

        return JsonResponse({"status": "success", "data": data})

    except MessageTemplates.DoesNotExist:
        return JsonResponse({"status": "error", "message": "Template not found"})
    except customer_details.DoesNotExist:
        return JsonResponse({"status": "error", "message": "Customer not found"})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})



# ============================================================
# 2️⃣ SEND WHATSAPP
# ============================================================
@csrf_exempt
def send_whatsapp_message_api(request, customer_id):
    try:
        if request.method != "POST":
            return JsonResponse({"status": "error", "message": "POST required"})

        template_id = request.POST.get("template_id")
        if not template_id:
            return JsonResponse({"status": "error", "message": "template_id missing"})

        template = MessageTemplates.objects.get(id=template_id)
        customer = customer_details.objects.get(id=customer_id)

        # Render message with dynamic fields
        rendered_body = render_dynamic_template(template.body, customer_id)

        status, resp = send_whatsapp(customer, rendered_body)

        return JsonResponse({
            "status": status,
            "response": resp,
            "body_sent": rendered_body,
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})



# ============================================================
# 3️⃣ SEND EMAIL
# ============================================================
@csrf_exempt
def send_email_message_api(request, customer_id):
    try:
        if request.method != "POST":
            return JsonResponse({"status": "error", "message": "POST required"})

        template_id = request.POST.get("template_id")
        if not template_id:
            return JsonResponse({"status": "error", "message": "template_id missing"})

        template = MessageTemplates.objects.get(id=template_id)
        customer = customer_details.objects.get(id=customer_id)

        rendered_subject = template.subject or ""
        rendered_body = render_dynamic_template(template.body, customer_id)

        status, resp = send_email(customer, rendered_subject, rendered_body)

        return JsonResponse({
            "status": status,
            "response": resp,
            "subject": rendered_subject,
            "body_sent": rendered_body,
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})



# ============================================================
# 4️⃣ RENDER MESSAGING PAGE
# ============================================================
def messaging_page(request):
    templates = MessageTemplates.objects.all()
    customers = customer_details.objects.all()

    return render(request, "crmapp/messaging_page.html", {
        "templates": templates,
        "customers": customers
    })

