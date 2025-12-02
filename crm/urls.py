"""
URL configuration for crm project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('crmapp.urls')),
    path('ocrapp/', include('ocrapp.urls')),
    path('open_ai/', include('open_ai.urls')),
    path('lead_automation/', include('lead_automation.urls')),
    path('generate_quotation/', include('generate_quotation.urls')),
    path('email_sender/', include('email_sender.urls')),
    path('schedule_meetings/', include('schedule_meetings.urls')),
    path('generate_invoice/', include('generate_invoice.urls')),
    path("chat_app/", include("chat_app.urls")),
    path("dashboard/", include("dashboard.urls")),
    path('api/', include('recommender.urls')),


    # AI Calling frontend
    path("ai-calling/", include("ai_calling.urls")),  # <--- add this

    # AI Calling API
    path("api/ai-calling/", include("ai_calling.api_urls")),
    path('', include('recommender.urls')),
    path('recommendations/', include('recommender.urls')),
    path("messages/", include("crmapp.urls_message")),


    


]
