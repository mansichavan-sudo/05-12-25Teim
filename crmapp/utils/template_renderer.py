from crmapp.models import customer_details, Product
from recommender.models import PestRecommendation

def render_dynamic_template(template_body, customer_id):
    try:
        customer = customer_details.objects.get(id=customer_id)
        rec = PestRecommendation.objects.filter(customer_id=customer_id).first()

        base_product = rec.base_product.name if rec else ""
        recommended_product = rec.recommended_product.name if rec else ""

        message = template_body.format(
            customer_name=customer.customer_name,
            product=base_product,
            recommended_product=recommended_product
        )
        return message

    except Exception as e:
        return f"{{ERROR: {e}}}"
