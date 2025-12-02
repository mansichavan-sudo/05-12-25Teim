def build_recommendation_message(customer, recommended_products):
    first_name = customer.customer_name.split(" ")[0]

    product_lines = "\n".join(
        [f"• {p.product_name} – ₹{p.price}" for p in recommended_products]
    )

    message = f"""
Hi {first_name},

Based on your pest problems, we have selected the best treatment products for you:

{product_lines}

Reply YES to confirm your service.
- TEIM Pest Control AI System
"""

    return message.strip()
