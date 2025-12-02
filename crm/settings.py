"""
Django settings for crm project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ------------------------------------------------------
# SECURITY
# ------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "fallback-secret-key")
DEBUG = os.environ.get("DEBUG", "False") == "True"
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "").split(",") if os.environ.get("ALLOWED_HOSTS") else []
 
INSTALLED_APPS = [
    # Django defaults
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Celery integrations
    'django_celery_beat',
    'django_celery_results',

    # Your apps
    'crmapp',   #  
    'schedule_meetings',
    'ocrapp',
    'email_sender',
    'generate_invoice',
    'chat_app',
    'dashboard',
    'recommender',
    'ai_calling',

    # REST Framework
    'rest_framework',
]



MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'crm.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'crm.wsgi.application'

# ------------------------------------------------------
# DATABASE
# ------------------------------------------------------
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'teim1',
        'USER': 'root',
        'PASSWORD': 'root',
        'HOST': 'localhost',
        'PORT': '3306',
    }

}

# ------------------------------------------------------
# PASSWORDS
# ------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ------------------------------------------------------
# INTERNATIONALIZATION
# ------------------------------------------------------
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

# ------------------------------------------------------
# STATIC & MEDIA
# ------------------------------------------------------
STATIC_URL = 'static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

SITE_URL = os.environ.get("SITE_URL", "https://www.teimcrm.com")
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ------------------------------------------------------
# AUTH
# ------------------------------------------------------
SECURITY_KEY = os.environ.get("SECURITY_KEY", "Seva@Facility@0000")

AUTHENTICATION_BACKENDS = [
    'crmapp.backends.ContactNumberBackend',
    'django.contrib.auth.backends.ModelBackend',
]

# ------------------------------------------------------
# DJANGO REST FRAMEWORK
# ------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
}

# ------------------------------------------------------
# CELERY
# ------------------------------------------------------
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "django-db")
CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
CELERY_CACHE_BACKEND = 'django-cache'
CELERY_BEAT_SCHEDULER = 'django_celery_beat.schedulers:DatabaseScheduler'

# ------------------------------------------------------
# TWILIO (AI CALLING)
# ------------------------------------------------------
USE_TWILIO = os.environ.get("USE_TWILIO", "False") == "True"
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_DEFAULT_CALLER = os.environ.get("TWILIO_DEFAULT_CALLER", "")

# ------------------------------------------------------
# VAPI (AI CALLING)
# ------------------------------------------------------
USE_VAPI = os.environ.get("USE_VAPI", "False") == "True"
VAPI_API_KEY = os.environ.get("VAPI_API_KEY", "")
VAPI_ASSISTANT_ID = os.environ.get("VAPI_ASSISTANT_ID", "")
VAPI_PHONE_NUMBER_ID = os.environ.get("VAPI_PHONE_NUMBER_ID", "")
VAPI_API_URL = os.environ.get("VAPI_API_URL", "https://api.vapi.ai")


# ------------------------------------------------------
# CRM API (OPTIONAL)
# ------------------------------------------------------
CRM_LEADS_ENDPOINT = os.environ.get("CRM_LEADS_ENDPOINT", "")
CRM_API_TOKEN = os.environ.get("CRM_API_TOKEN", "")

# ------------------------------------------------------
# UPLOAD LIMITS
# ------------------------------------------------------
DATA_UPLOAD_MAX_MEMORY_SIZE = int(os.environ.get("DATA_UPLOAD_MAX_MEMORY_SIZE", 1073741824))  # 1GB
FILE_UPLOAD_MAX_MEMORY_SIZE = int(os.environ.get("FILE_UPLOAD_MAX_MEMORY_SIZE", 1073741824))  # 1GB

# ------------------------------------------------------
# EMAIL
# ------------------------------------------------------
# ------------------------------------------------------
# EMAIL (Gmail SMTP)
# ------------------------------------------------------
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'  # MUST be SMTP
# EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"  # ❌ REMOVE THIS

EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True

EMAIL_HOST_USER = "mansiichavan2004@gmail.com"
EMAIL_HOST_PASSWORD = "evoypskskoxnpxiw"   # Gmail App Password

DEFAULT_FROM_EMAIL = EMAIL_HOST_USER

# ------------------------------------------------------
# WHATSAPP
# ------------------------------------------------------
WHATSAPP_API = os.environ.get("WHATSAPP_API", "")
WHATSAPP_CHANNEL_ID = os.environ.get("WHATSAPP_CHANNEL_ID", "")
 
# ------------------------------------------------------
# WHATSAPP (RapBooster)
# ------------------------------------------------------
RAPBOOSTER_API_URL = "https://api.rapbooster.com/v1/send"
RAPBOOSTER_API_KEY = "6538c8eff027d41e9151"

#FAST2SMS_API_KEY = "lmECgGI0f57i2x94H81uqVtTyObKhzFZMNLXA3oapseB6RQcJD5ZsIrukQRLzoCcKHWaBF6TliSXdUgA"
