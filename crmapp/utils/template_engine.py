from crmapp.models import customer_details, Product
from recommender.models import PestRecommendation

def render_recommendation_message(customer_id, template):
    customer = customer_details.objects.get(id=customer_id)

    # Fetch latest recommendation
    rec = PestRecommendation.objects.filter(customer_id=customer_id).order_by('-created_at').first()

    base_product_name = Product.objects.get(product_id=rec.base_product_id).product_name
    recommended_product_name = Product.objects.get(product_id=rec.recommended_product_id).product_name

    rendered_body = template.body.format(
        customer_name = customer.fullname,
        product_name = base_product_name,
        recommended_product = recommended_product_name,
        score = rec.confidence_score
    )

    return rendered_body, base_product_name, recommended_product_name
