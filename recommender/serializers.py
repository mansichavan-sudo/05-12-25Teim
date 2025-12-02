# recommendations/serializers.py  (or crmapp/serializers.py)
from rest_framework import serializers
from crmapp.models import customer_details

class CustomerSerializer(serializers.ModelSerializer):
    class Meta:
        model = customer_details
        fields = [
            "id",
            "fullname",
            "primarycontact",
            "secondarycontact",
        ]
