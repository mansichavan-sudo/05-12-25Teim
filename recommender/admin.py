from django.contrib import admin
from .models import Item, Rating, SavedModel, Interaction, PestRecommendation
from crmapp.models import SentMessageLog  # only CRM log model

 
# Register recommender app models
admin.site.register(Item)
admin.site.register(Rating)
admin.site.register(SavedModel)
admin.site.register(Interaction)
admin.site.register(PestRecommendation)

# Register CRM logs separately if needed
# admin.site.register(MessageTemplates)
# admin.site.register(CrmSentLog)
