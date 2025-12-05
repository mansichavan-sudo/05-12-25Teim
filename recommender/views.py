# recommender/views.py

from django.shortcuts import render
from django.db import connection
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
import json
import pickle
import os
import requests
import re
from .rapbooster_api import send_whatsapp_message, send_email_message, send_recommendation_message
from django.shortcuts import get_object_or_404
# Models
from crmapp.models import MessageTemplates, Product, customer_details ,PurchaseHistory
from .models import Item, Rating, PestRecommendation

# Recommender Engine
from .recommender_engine import (
    get_content_based_recommendations,
    get_collaborative_recommendations,
    get_upsell_recommendations,
    get_crosssell_recommendations,
    generate_recommendations_for_user,
    get_user_based_recommendations
)

from .utils import send_recommendation_message

# Helper: Render placeholders
def render_template(text, data):
    return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", lambda m: str(data.get(m.group(1), "")), text or "")

# ============================================================
# 1️⃣ RECOMMENDATION UI
# ============================================================
def recommendation_ui(request):
    customers = customer_details.objects.all().values("id", "fullname")
    templates = MessageTemplates.objects.filter(is_active=True).order_by("category", "message_type")
    
    selected_product_id = request.GET.get("product_id")
    customer_id = request.GET.get("customer_id")

    customer = None
    product = None
    
    if selected_product_id:
        try:
            product = Product.objects.get(id=selected_product_id)
        except Product.DoesNotExist:
            pass
    
    if customer_id:
        try:
            customer = customer_details.objects.get(id=customer_id)
            if getattr(customer, 'lead_status', None):
                templates = MessageTemplates.objects.filter(
                    category='lead', 
                    lead_status=customer.lead_status, 
                    is_active=True
                ).order_by("message_type")
            else:
                templates = MessageTemplates.objects.filter(
                    category='lead', 
                    is_active=True
                ).order_by("message_type")
        except customer_details.DoesNotExist:
            pass
    
    return render(request, 'recommender/recommendations_ui.html', {
        'templates': templates,
        'customer': customer,
        'product': product,
        'customers': customers,
    })

# ============================================================
# 2️⃣ CONTENT-BASED RECOMMENDATIONS (product → product)
# ============================================================
def recommendations_view(request):
    product_name = request.GET.get('product')
    if not product_name:
        return JsonResponse({'error': 'Please provide a product name.'}, status=400)
    try:
        results = get_content_based_recommendations(product_name)
        return JsonResponse({'recommended_products': results})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# ============================================================
# 3️⃣ COLLABORATIVE FILTERING (customer → customer)
# ============================================================
def collaborative_view(request, customer_id):
    try:
        results = get_collaborative_recommendations(customer_id)
        return JsonResponse({'similar_customers': results})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# ============================================================
# 4️⃣ UPSELL (product → higher product)
# ============================================================
def upsell_view(request, product_id):
    try:
        results = get_upsell_recommendations(product_id)
        return JsonResponse({'upsell_suggestions': results})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# ============================================================
# 5️⃣ CROSS-SELL (customer → new related products)
# ============================================================
def crosssell_view(request, customer_id):
    try:
        results = get_crosssell_recommendations(customer_id)
        return JsonResponse({'cross_sell_suggestions': results})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
# ============================================================
# 6️⃣ FINAL DASHBOARD TABLE (CRM)
# ============================================================
def recommendation_dashboard(request):
    try:
        filter_type = (request.GET.get("type") or "").strip().lower()
        search = request.GET.get("search", "").strip()
        sort_column = request.GET.get("sort", "confidence_score")
        sort_order = request.GET.get("order", "desc")

        valid_columns = {
            "customer_name": "c.fullname",
            "phone": "c.primarycontact",
            "purchase_product": "ph.product_name",
            "purchase_date": "ph.purchased_at",
            "price": "ph.total_amount",
            "recommended_product": "rp.product_name",
            "recommended_category": "rp.category",
            "recommendation_type": "pr.recommendation_type",
            "confidence_score": "pr.confidence_score",
        }

        sort_field = valid_columns.get(sort_column, "pr.confidence_score")
        order_sql = "DESC" if sort_order.lower() == "desc" else "ASC"

        sql = """
            SELECT 
                c.id AS customer_id,
                c.fullname AS customer_name,
                c.primarycontact AS phone,

                CONCAT_WS(', ',
                    c.soldtopartyaddress,
                    c.soldtopartycity,
                    c.soldtopartystate,
                    c.soldtopartypostal
                ) AS full_address,

                ph.product_name AS purchase_product,
                ph.quantity AS quantity,
                ph.total_amount AS price,
                ph.purchased_at AS purchase_date,

                rp.product_name AS recommended_product,
                rp.category AS recommended_category,
                pr.recommendation_type,
                pr.confidence_score

            FROM crmapp_customer_details c
            LEFT JOIN crmapp_purchasehistory ph ON c.id = ph.customer_id
            LEFT JOIN pest_recommendations pr ON c.id = pr.customer_id
            LEFT JOIN crmapp_product rp ON rp.product_id = pr.recommended_product_id
            WHERE 1 = 1
        """

        params = []

        if filter_type:
            sql += " AND LOWER(pr.recommendation_type) LIKE %s"
            params.append(f"%{filter_type}%")

        if search:
            sql += """
                AND (
                    c.fullname LIKE %s OR
                    ph.product_name LIKE %s OR
                    rp.product_name LIKE %s
                )
            """
            params.extend([f"%{search}%"] * 3)

        sql += f" ORDER BY {sort_field} {order_sql};"

        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        data = []
        for row in rows:
            data.append({
                "customer_id": row[0],
                "customer_name": row[1],
                "phone": row[2],
                "address": row[3],

                # Fallback values for null purchase history
                "purchase_product": row[4] or "—",
                "quantity": row[5] or "—",
                "price": float(row[6]) if row[6] is not None else "—",
                "purchase_date": row[7] or "—",

                "recommended_product": row[8] or "—",
                "recommended_category": row[9] or "—",
                "recommendation_type": row[10] or "—",
                "confidence_score": float(row[11]) if row[11] else "—",
            })

        page_obj = Paginator(data, 15).get_page(request.GET.get("page"))

        return render(request, "recommender/recommendation_dashboard.html", {
            "recommendations": page_obj.object_list,
            "page_obj": page_obj,
            "sort": sort_column,
            "order": sort_order,
            "filter_type": filter_type,
            "search": search,
        })

    except Exception as e:
        return render(request, "recommender/recommendation_dashboard.html", {
            "recommendations": [],
            "error": str(e),
        })

# ============================================================
# 7️⃣ GET ALL PRODUCTS
# ============================================================
def get_all_products(request):
    try:
        products = list(Product.objects.values_list("product_name", flat=True))
        return JsonResponse({'products': products})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


import logging
logger = logging.getLogger(__name__)

@csrf_exempt
@login_required
def api_ai_personalized(request, customer_id):
    try:
        customer_id = int(customer_id)
        customer = get_object_or_404(customer_details, id=customer_id)

        # Safe Recommendations
        try:
            recommendations = generate_recommendations_for_user(
                customer_id=customer_id,
                top_n=5
            )
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations = []

        results = []

        for r in recommendations:
            try:
                # CASE 1: Item model returned
                if isinstance(r, Item):
                    results.append({
                        "product_id": r.product_id,
                        "title": r.title,
                        "category": r.category,
                        "tags": r.tags,
                        "confidence_score": getattr(r, "score", None),
                    })

                # CASE 2: Product model returned
                elif hasattr(r, "product_name"):
                    results.append({
                        "product_id": r.id,
                        "title": r.product_name,
                        "category": getattr(r, "category", None),
                        "tags": getattr(r, "tags", ""),
                        "confidence_score": getattr(r, "score", None),
                    })

                # Fallback
                else:
                    results.append({
                        "product_id": getattr(r, "id", None),
                        "title": getattr(r, "title", "Unknown Item"),
                        "category": getattr(r, "category", None),
                        "tags": getattr(r, "tags", None),
                        "confidence_score": getattr(r, "score", None),
                    })
            except Exception as e:
                logger.warning(f"Recommendation item processing error: {e}")
                continue

        return JsonResponse({
            "customer_id": customer.id,
            "customer_name": customer.fullname,
            "recommendations": results,
        })

    except ValueError:
        return JsonResponse({"error": "Invalid customer ID"}, status=400)
    except customer_details.DoesNotExist:
        return JsonResponse({"error": "Customer not found"}, status=404)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JsonResponse({"error": "Internal server error"}, status=500)



# ============================================================
# 9️⃣ AUTOMATIC MESSAGE GENERATION + SEND
# ============================================================
@csrf_exempt
def generate_message(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    try:
        data = json.loads(request.body)

        customer = data.get("customer_name")
        base = data.get("base_product")
        rec = data.get("recommended_product")
        rec_type = data.get("recommendation_type")
        phone_number = data.get("phone_number")

        if not all([customer, base, rec, rec_type, phone_number]):
            return JsonResponse({"error": "Missing fields"}, status=400)

        message = (
            f"Hello {customer}, we recommend trying our {rec} as a perfect "
            f"{rec_type.lower()} option with your {base}. "
            f"It ensures better pest control results! 🌾🛡️"
        )

        from .rapbooster_api import send_recommendation_message

        status_code, api_response = send_recommendation_message(
            phone_number=phone_number,
            message=message,
            customer_name=customer
        )

        return JsonResponse({
            "customer": customer,
            "phone": phone_number,
            "message": message,
            "status": "sent" if status_code == 200 else "failed",
            "api_response": api_response
        })

    except Exception as e:
        logger.error(f"Message generation error: {e}")
        return JsonResponse({"error": str(e)}, status=500)



# ============================================================
# 🔟 RAP BOOSTER MESSAGE SENDER (FINAL FIXED VERSION)
# ============================================================
@csrf_exempt
def send_message_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        # ENV fallback
        RAPBOOSTER_API_KEY = os.getenv("RAPBOOSTER_API_KEY")
        if not RAPBOOSTER_API_KEY:
            RAPBOOSTER_API_KEY = "6538c8eff027d41e9151"  # Safe fallback

        RAPBOOSTER_SEND_URL = "https://rapbooster.in/api/v1/sendMessage"

        data = json.loads(request.body)
        template_id = data.get("template_id")
        customer_id = data.get("customer_id")

        if not template_id or not customer_id:
            return JsonResponse({"error": "template_id and customer_id are required"}, status=400)

        template = MessageTemplates.objects.get(id=template_id)
        customer = customer_details.objects.get(id=customer_id)

        phone = str(customer.primarycontact).strip()

        # Phone validation: 10–15 digits
        if not re.fullmatch(r"\+?\d{10,15}", phone):
            return JsonResponse({"error": "Invalid phone number format"}, status=400)

        # Render the template body
        rendered_body = render_template(template.body, {
            "customer_name": customer.fullname,
            "recommended_product": data.get("recommended_product", "")
        })

        payload = {
            "apikey": RAPBOOSTER_API_KEY,
            "phone": phone,
            "message": rendered_body
        }

        response = requests.post(RAPBOOSTER_SEND_URL, json=payload, timeout=10)

        try:
            resp_json = response.json()
        except:
            resp_json = {}

        # Success check
        success = response.status_code == 200 and resp_json.get("status") == "success"

        SentMessageLog.objects.create(
            template=template,
            recipient=phone,
            channel=template.message_type,
            rendered_body=rendered_body,
            status="success" if success else "failed",
            provider_response=response.text,
        )

        if success:
            return JsonResponse({
                "status": "success",
                "message_id": resp_json.get("message_id"),
                "phone": resp_json.get("phone")
            })

        return JsonResponse({
            "status": "failed",
            "error": resp_json.get("error", "Unknown error"),
            "http_status": response.status_code
        }, status=500)

    except MessageTemplates.DoesNotExist:
        return JsonResponse({"error": "Template not found"}, status=404)

    except customer_details.DoesNotExist:
        return JsonResponse({"error": "Customer not found"}, status=404)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    except requests.RequestException as e:
        logger.error(f"RapBooster error: {e}")
        return JsonResponse({"error": "Network error contacting RapBooster"}, status=500)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JsonResponse({"error": "Internal server error"}, status=500)

# ============================================================
# 1️⃣1️⃣ CUSTOMER RECOMMENDATIONS API
# ============================================================
def customer_recommendations_api(request, customer_id):
    try:
        recommendations = get_user_based_recommendations(customer_id)
        return JsonResponse({
            "customer_id": customer_id,
            "recommendations": recommendations
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def get_all_customers(request):
    customers = customer_details.objects.all()

    data = [
        {
            "customer_id": c.id,
            "customer_name": c.fullname,
            "primarycontact": str(c.primarycontact) if c.primarycontact else "",
            "secondarycontact": str(c.secondarycontact) if c.secondarycontact else "",
            "phone": str(c.primarycontact or c.secondarycontact or ""),
        }
        for c in customers
    ]

    return JsonResponse({"customers": data})


# ============================================================
# 1️⃣3️⃣ GET CUSTOMER PHONE
# ============================================================
def customer_phone(request, cid):
    customer = customer_details.objects.filter(id=cid).first()
    if customer:
        return JsonResponse({"phone": customer.primarycontact})
    return JsonResponse({"phone": None})

# ============================================================
# 1️⃣4️⃣ MESSAGE LOG VIEW
# ============================================================
def message_log_view(request):
    logs = SentMessageLog.objects.all().order_by('-created_at')[:100]
    return render(request, 'recommender/message_logs.html', {'logs': logs})

# ============================================================
# SEND MESSAGE API (Final RapBooster Version)
# ============================================================ 
import json, os, requests


# -----------------------------------------
# 🔧 Placeholder replace function
# -----------------------------------------
def replace_placeholders(message, values: dict):
    """Replaces {{keys}} in the message using values dict."""
    if not message:
        return message

    for key, val in values.items():
        placeholder = "{{" + key + "}}"
        message = message.replace(placeholder, str(val))

    return message


# -----------------------------------------
# 📩 Main WhatsApp Send API
# -----------------------------------------
@csrf_exempt
def send_message_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode())

        # Frontend params
        customer_name = data.get("customer_name")
        phone_number = data.get("phone_number")
        message = data.get("message")

        # Extra variables (optional)
        extra = data.get("extra", {})  
        # Example: { "product": "Cockroach Control", "due_date": "5 Dec" }

        # -----------------------------
        # 🛑 Validation
        # -----------------------------
        if not phone_number:
            return JsonResponse({"error": "Phone number missing"}, status=400)
        if not message:
            return JsonResponse({"error": "Message missing"}, status=400)

        # -----------------------------
        # 🔄 Placeholder Replacement
        # -----------------------------
        final_message = replace_placeholders(
            message,
            {
                "customer_name": customer_name,
                "phone": phone_number,
                **extra  # add all dynamic variables
            }
        )

        # -----------------------------
        # 📡 RapBooster API Call
        # -----------------------------
        RAPBOOSTER_API_KEY = os.getenv("RAPBOOSTER_API_KEY") or "6538c8eff027d41e9151"
        RAPBOOSTER_URL = "https://api.rapbooster.com/v1/send"

        payload = {
            "apikey": RAPBOOSTER_API_KEY,
            "phone": phone_number,
            "message": final_message
        }

        response = requests.post(RAPBOOSTER_URL, json=payload, timeout=10)

        try:
            provider = response.json()
        except:
            provider = {"raw": response.text}

        success = (
            response.status_code == 200 and
            provider.get("status") == "success"
        )

        # -----------------------------
        # ✅ SUCCESS RESPONSE
        # -----------------------------
        if success:
            return JsonResponse({
                "status": "success",
                "customer": customer_name,
                "phone": phone_number,
                "sent_message": final_message,
                "rapbooster_message_id": provider.get("message_id"),
                "queue_status": provider.get("queue_status"),
            })

        # -----------------------------
        # ❌ FAILED
        # -----------------------------
        return JsonResponse({
            "status": "failed",
            "http_code": response.status_code,
            "sent_message": final_message,
            "provider_response": provider,
        }, status=400)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


from .rapbooster_api import send_whatsapp_message, send_email_message
from django.conf import settings
from django.core.mail import send_mail

from crmapp.models import SentMessageLog
from .rapbooster_api import send_recommendation_message
@csrf_exempt
def send_whatsapp(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=400)

    phone = request.POST.get("phone")
    message = request.POST.get("message")
    customer_id = request.POST.get("customer_id")

    from crmapp.models import customer_details
    customer = customer_details.objects.filter(id=customer_id).first()

    status, provider_response = send_whatsapp_message(customer, message)

    return JsonResponse({
        "status": status,
        "provider_response": provider_response
    })

@csrf_exempt
def send_email(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    email = request.POST.get("email")
    subject = request.POST.get("subject", "")
    message = request.POST.get("message", "")
    customer_id = request.POST.get("customer_id")

    from crmapp.models import customer_details
    customer = customer_details.objects.filter(id=customer_id).first()

    status, provider_response = send_email_message(customer, subject, message)

    return JsonResponse({
        "status": status,
        "provider_response": provider_response
    })
 
from crmapp.models import customer_details

def api_customers(request):
    customers = customer_details.objects.all()

    data = []
    for c in customers:
        primary = c.primarycontact if c.primarycontact else ""
        secondary = c.secondarycontact if c.secondarycontact else ""

        # Final fallback phone
        final_phone = ""
        if primary:
            final_phone = str(primary)
        elif secondary:
            final_phone = str(secondary)

        data.append({
            "customer_id": c.id,
            "customer_name": c.fullname,

            # Send all phone fields
            "primarycontact": str(primary),
            "secondarycontact": str(secondary),
            "phone": final_phone,     # REQUIRED
        })

    return JsonResponse({"customers": data})


@csrf_exempt
def rapbooster_webhook(request):
    # Allow browser GET testing
    if request.method == "GET":
        return JsonResponse({"message": "RapBooster Webhook Active", "method": "GET"})

    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    try:
        data = json.loads(request.body.decode())

        message_id = data.get("message_id")
        status = data.get("status")

        if not message_id:
            return JsonResponse({"error": "message_id missing"}, status=400)

        log = SentMessageLog.objects.filter(message_id=message_id).first()

        if not log:
            return JsonResponse({"error": "No log found for message_id"}, status=404)

        status_mapping = {
            "sent": "sent",
            "delivered": "delivered",
            "read": "read",
            "failed": "failed",
            "queued": "queued"
        }

        log.status = status_mapping.get(status, "sent")
        log.provider_response = json.dumps(data)
        log.save()

        return JsonResponse({"success": True})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

from crmapp.models import SentMessageLog, customer_details

def message_timeline_api(request, customer_id):
    logs = SentMessageLog.objects.filter(customer_id=customer_id).order_by("-created_at")

    data = []
    for log in logs:
        data.append({
            "id": log.id,
            "recipient": log.recipient,
            "channel": log.channel,
            "rendered_subject": log.rendered_subject,
            "rendered_body": log.rendered_body,
            "status": log.status,
            "message_id": log.message_id,
            "provider_response": log.provider_response,
            "created_at": log.created_at.strftime("%Y-%m-%d %H:%M"),
            "updated_at": log.updated_at.strftime("%Y-%m-%d %H:%M"),
        })

    return JsonResponse({"timeline": data})


def get_purchase_history(customer_id):
    history = []

    # ------------------------------
    # 1. ServiceProduct (FK → Product) ✔ MOST ACCURATE
    # ------------------------------
    from crmapp.models import ServiceProduct
    sp = ServiceProduct.objects.filter(service__custid_id=customer_id)
    for s in sp:
        history.append({
            'product_id': s.product_id,
            'product_name': s.product.product_name,
            'date': s.service.date if hasattr(s.service, 'date') else None,
            'source': 'service_product'
        })

    # ------------------------------
    # 2. Invoice (NO FK) → match product_name to Product table
    # ------------------------------
    from crmapp.models import invoice, Product

    inv = invoice.objects.filter(custid_id=customer_id)
    for i in inv:
        product_name = i.description_of_goods.strip()
        try:
            match = Product.objects.get(product_name__icontains=product_name)
            history.append({
                'product_id': match.product_id,
                'product_name': match.product_name,
                'date': i.invoice_date,
                'source': 'invoice'
            })
        except Product.DoesNotExist:
            continue

    return history



def api_purchase_history(request, cid):
    try:
        customer = customer_details.objects.get(customer_id=cid)
    except customer_details.DoesNotExist:
        return JsonResponse({"error": "Invalid customer"}, status=404)

    history = PurchaseHistory.objects.filter(customer=customer).order_by("-purchased_at")

    data = []
    for p in history:
        data.append({
            "product": p.product.product_name if p.product else p.product_name,
            "quantity": float(p.quantity),
            "total_amount": float(p.total_amount),
            "date": p.purchased_at.strftime("%d-%m-%Y %I:%M %p"),
        })

    return JsonResponse({"history": data})