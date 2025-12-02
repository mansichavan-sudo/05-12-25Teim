from django.db import models
from crmapp.models import Product, customer_details, MessageTemplates

# ---------------------------------------------------
# ITEM TABLE → connected to Product
# ---------------------------------------------------
class Item(models.Model):
    id = models.BigAutoField(primary_key=True)
    title = models.CharField(max_length=255)
    description = models.TextField()
    category = models.CharField(max_length=128)
    tags = models.CharField(max_length=512)
    created_at = models.DateTimeField(auto_now_add=True)

    product = models.OneToOneField(
        Product,
        on_delete=models.CASCADE,
        db_column='product_id',
        related_name='item',
        null=True,
        blank=True
    )

    class Meta:
        db_table = 'recommender_item'

    def __str__(self):
        return self.title


# ---------------------------------------------------
# RATING TABLE (customer + product)
# ---------------------------------------------------
class Rating(models.Model):
    id = models.BigAutoField(primary_key=True)
    rating = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    product = models.ForeignKey(
        Product,
        on_delete=models.CASCADE,
        db_column='product_id',
        related_name='rating_products',   # FIXED
        null=True,
        blank=True
    )

    customer = models.ForeignKey(
        customer_details,
        on_delete=models.CASCADE,
        db_column='customer_id',
        related_name='rating_customers',  # FIXED
        null=True,
        blank=True
    )

    class Meta:
        db_table = 'recommender_rating'

    def __str__(self):
        return f"{self.customer.fullname if self.customer else 'Unknown'} → {self.product.product_name if self.product else 'Unknown'}: {self.rating}"


# =========================================================
# Saved ML models
# =========================================================
class SavedModel(models.Model):
    name = models.CharField(max_length=100, unique=True)
    file_path = models.CharField(max_length=512)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


# =========================================================
# INTERACTIONS LOG
# ========================================================= 
class Interaction(models.Model):
    INTERACTION_TYPES = [
        ('view', 'View'),
        ('click', 'Click'),
        ('purchase', 'Purchase'),
        ('call', 'Call'),
        ('recommend', 'Recommendation Shown'),
    ]

    customer = models.ForeignKey(
        customer_details,
        on_delete=models.CASCADE,
        related_name='interaction_customers',
        null=True,      # ADD THIS
        blank=True      # ADD THIS
    )

    product = models.ForeignKey(
        Product,
        on_delete=models.CASCADE,
        related_name='interaction_products',
        null=True,      # ADD THIS
        blank=True      # ADD THIS
    )

    interaction_type = models.CharField(max_length=50, choices=INTERACTION_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(blank=True, null=True)

    class Meta:
        db_table = 'recommender_interaction'
        unique_together = ('customer', 'product', 'interaction_type')
        ordering = ['-timestamp']

# =========================================================
# FINAL RECOMMENDATIONS TABLE
# ========================================================= 
# recommender/models.py
 

# ... (other models unchanged) ...


class PestRecommendation(models.Model):
    # Use canonical keys (no spaces/hyphens, lowercase)
    CANONICAL_TYPES = {
        'upsell': 'upsell',
        'up-sell': 'upsell',
        'up sell': 'upsell',
        'up_sell': 'upsell',
        'crosssell': 'crosssell',
        'cross-sell': 'crosssell',
        'cross sell': 'crosssell',
        'cross_sell': 'crosssell',
        'content': 'content',
        'content-based': 'content',
        'content based': 'content',
        'collaborative': 'collaborative'
    }

    RECOMMENDATION_TYPES_CHOICES = [
        ('upsell', 'Upsell'),
        ('crosssell', 'Cross-sell'),
        ('content', 'Content-Based'),
        ('collaborative', 'Collaborative'),
    ]

    customer = models.ForeignKey(
        customer_details,
        on_delete=models.CASCADE,
        related_name='pest_recommendations',
        null=True,
        blank=True
    )
    base_product = models.ForeignKey(
        Product,
        on_delete=models.CASCADE,
        related_name='base_pest_recommendations',
        null=True,
        blank=True
    )
    recommended_product = models.ForeignKey(
        Product,
        on_delete=models.CASCADE,
        related_name='recommended_pest',
        null=True,
        blank=True
    )

    recommendation_type = models.CharField(
        max_length=20,
        choices=RECOMMENDATION_TYPES_CHOICES,
        null=True,
        blank=True
    )
    confidence_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'pest_recommendations'

    def __str__(self):
        return f"{self.get_recommendation_type_display() if self.recommendation_type else 'Unknown'} for {self.customer.fullname if self.customer else 'Unknown'}"

    @staticmethod
    def normalize_recommendation_type(val: str):
        """Return canonical key or None."""
        if not val:
            return None
        v = str(val).strip().lower()
        # direct exact match
        if v in PestRecommendation.CANONICAL_TYPES:
            return PestRecommendation.CANONICAL_TYPES[v]
        # try common normalizations
        simple = v.replace('-', '').replace(' ', '').replace('_', '')
        if simple in PestRecommendation.CANONICAL_TYPES:
            return PestRecommendation.CANONICAL_TYPES[simple]
        # fallback: try to match substring
        if 'up' in simple and 'sell' in simple:
            return 'upsell'
        if 'cross' in simple and 'sell' in simple:
            return 'crosssell'
        if 'content' in simple:
            return 'content'
        if 'collaborative' in simple:
            return 'collaborative'
        return None

    def save(self, *args, **kwargs):
        # Normalize recommendation_type to canonical key before save
        norm = PestRecommendation.normalize_recommendation_type(self.recommendation_type)
        if norm:
            self.recommendation_type = norm
        else:
            # keep None / blank if unknown
            self.recommendation_type = None
        super().save(*args, **kwargs)




class HybridRankingDebug(models.Model):
    customer = models.ForeignKey(
        customer_details,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    generated_at = models.DateTimeField(auto_now_add=True)
    num_candidates = models.IntegerField(default=0)
    debug_log = models.JSONField(null=True, blank=True)

    class Meta:
        db_table = "hybrid_ranking_debug"

    def __str__(self):
        return f"Hybrid Debug - {self.customer_id} ({self.generated_at})"
