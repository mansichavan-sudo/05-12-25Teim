from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import (
    PaymentsRecord,
    UserProfile,
    service_management,
    ServiceProduct,
    MessageTemplates,
)

from django.contrib import admin
from .models import SentMessageLog


@admin.register(SentMessageLog)
class SentMessageLogAdmin(admin.ModelAdmin):

    list_display = (
        'id',
        'customer',
        'recipient',
        'channel',
        'status',
        'created_at',   # FIXED
    )

    list_filter = (
        'channel',
        'status',
        'created_at',   # FIXED
    )

    search_fields = (
        'recipient',
        'message_id',
        'rendered_body',
    )

    ordering = ('-created_at',)

# ======================================
# 🔹 PaymentsRecord Admin
# ======================================
@admin.register(PaymentsRecord)
class PaymentsRecordAdmin(admin.ModelAdmin):
    list_display = [
        'payment_invoice_no',
        'main_invoice',
        'amount_paid',
        'amount_remaining',
        'payment_date',
        'ageing',
    ]
    readonly_fields = ['ageing']


# ======================================
# 🔹 Inline UserProfile in User Admin
# ======================================
class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profile'


# ======================================
# 🔹 Custom Filter for UserProfile.role
# ======================================
class RoleListFilter(admin.SimpleListFilter):
    title = 'Role'
    parameter_name = 'role'

    def lookups(self, request, model_admin):
        roles = UserProfile.objects.values_list('role', flat=True).distinct()
        return [(r, r) for r in roles if r]

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(userprofile__role=self.value())
        return queryset


# ======================================
# 🔹 Custom User Admin
# ======================================
class UserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)
    list_display = (
        'username',
        'email',
        'first_name',
        'last_name',
        'is_staff',
        'get_role',
        'get_phone',
    )
    list_filter = (
        'is_staff',
        'is_superuser',
        'is_active',
        'groups',
        RoleListFilter,  # ✅ Safe custom role filter
    )

    def get_role(self, obj):
        return getattr(obj.userprofile, 'role', None)
    get_role.short_description = 'Role'

    def get_phone(self, obj):
        return getattr(obj.userprofile, 'phone', None)
    get_phone.short_description = 'Phone'


# ======================================
# 🔹 Register all models
# ======================================
admin.site.unregister(User)
admin.site.register(User, UserAdmin)
admin.site.register(service_management)
admin.site.register(ServiceProduct)
admin.site.register(MessageTemplates)
