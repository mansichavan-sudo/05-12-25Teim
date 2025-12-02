# recommender/urls.py

from django.urls import path
from . import views
from .views import api_ai_personalized,message_timeline_api

urlpatterns = [

    # ======================================
    # 🌐 MAIN UI PAGES (prefix handled in crm/urls.py)
    # Final URLs: /recommendations/ui/, /recommendations/dashboard/, /recommendations/message-logs/
    # ======================================
    path('ui/', views.recommendation_ui, name='recommendation_ui'),
    path('dashboard/', views.recommendation_dashboard, name='recommendation_dashboard'),
    path('message-logs/', views.message_log_view, name='message_log_view'),

    # ======================================
    # 🧠 CORE PRODUCT & CUSTOMER API
    # Final URLs: /recommendations/api/products/, etc.
    # ======================================
    path('api/products/', views.get_all_products, name='get_all_products'),
    path('api/customers/', views.get_all_customers, name='get_all_customers'),
    path('api/customer/<int:cid>/phone/', views.customer_phone, name='customer_phone'),

    # ======================================
    # 🧠 AI & RECOMMENDATION API
    # Final URLs: /recommendations/api/recommendations/
    # ======================================
    path('api/recommendations/', views.recommendations_view, name='api_recommendations'),
    path('api/recommendations/<int:customer_id>/', api_ai_personalized, name='api_recommendations_customer'),
    path('api/user_recommendations/<int:customer_id>/', api_ai_personalized, name='api_user_recommendations'),
    path('api/customer_recommendations/<int:customer_id>/', views.customer_recommendations_api, name='customer_recommendations_api'),

    # Collaborative / Upsell / Cross-sell
    path('api/collaborative/<int:customer_id>/', views.collaborative_view, name='api_collaborative'),
    path('api/upsell/<int:product_id>/', views.upsell_view, name='api_upsell'),
    path('api/crosssell/<int:customer_id>/', views.crosssell_view, name='api_crosssell'),

    # ======================================
    # 💬 MESSAGE CREATION & SENDING
    path('api/generate-message/', views.generate_message, name='api_generate_message'),
    path('api/send-message/', views.send_message_api, name='api_send_message'),

    # ======================================
    # 📩 DIRECT WHATSAPP & EMAIL SENDERS
    # Final URLs: /recommendations/send_whatsapp/, /recommendations/send_email/
    path('send_whatsapp/', views.send_whatsapp, name='send_whatsapp'),
    path('send_email/', views.send_email, name='send_email'),

    path('webhook/rapbooster/', views.rapbooster_webhook, name='rapbooster_webhook'),
    path("timeline/<int:customer_id>/", message_timeline_api, name="timeline_api"),


]
