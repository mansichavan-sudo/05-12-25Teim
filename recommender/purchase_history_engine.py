from crmapp.models import Product, customer_details,PurchaseHistory
from django.db.models import Count

# ----------------------------------------------------
# USER PURCHASE HISTORY → SIMILAR CUSTOMERS → PRODUCTS
# ----------------------------------------------------
def history_recommend(customer_id, top_n=6):
    purchases = PurchaseHistory.objects.filter(customer_id=customer_id)

    if not purchases:
        return popular_products(top_n)

    bought_ids = list(purchases.values_list("product_id", flat=True))

    similar_customers = PurchaseHistory.objects.filter(
        product_id__in=bought_ids
    ).exclude(customer_id=customer_id)

    suggestions = similar_customers.exclude(product_id__in=bought_ids)\
                    .values("product_id")\
                    .annotate(score=Count("product_id"))\
                    .order_by("-score")[:top_n]

    results = []
    for s in suggestions:
        try:
            prod = Product.objects.get(product_id=s["product_id"])
            prod.score = float(s["score"])
            results.append(prod)
        except:
            pass

    return results


# ----------------------------------------------------
# FALLBACK IF USER HAS NO HISTORY
# ----------------------------------------------------
def popular_products(top_n=6):
    popular = PurchaseHistory.objects.values("product_id")\
                                     .annotate(score=Count("product_id"))\
                                     .order_by("-score")[:top_n]

    result = []
    for p in popular:
        try:
            prod = Product.objects.get(product_id=p["product_id"])
            prod.score = float(p["score"])
            result.append(prod)
        except:
            pass
    return result
