from datetime import date
from django.db import models
import random
from django.contrib.auth.models import User
from django.utils import timezone
from num2words import num2words # type: ignore
from django.contrib.auth.models import User
from pydantic import ValidationError


def generate_customerid():
    random_number = str(random.randint(1000, 9999))
    return f"DEFAULT{random_number}"




class UserProfile(models.Model):
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('sales', 'Sales'),
        ('technician', 'Technician'),
        ('customer', 'Customer'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    phone = models.CharField(max_length=15, blank=True, null=True)  

    def __str__(self):
        return f"{self.user.username} - {self.role}"

class SalesPerson(models.Model):
    full_name = models.CharField(max_length=100)
    date_of_joining = models.DateField()
    mobile_no = models.CharField(max_length=15)
    email = models.EmailField(unique=True)
    date_of_birth = models.DateField()

    def __str__(self):
        return self.full_name


from django.db import models
class QuotationTerm(models.Model):
    description = models.TextField()

    def __str__(self):
        return self.description[:50]

class InvoiceTerm(models.Model):
    description = models.TextField()




class customer_details(models.Model):
   
    fullname = models.CharField(max_length=100)
    primaryemail=models.EmailField()
    secondaryemail=models.EmailField(null=True , blank=True)
    primarycontact=models.BigIntegerField( unique=True)
    secondarycontact=models.BigIntegerField(null=True , blank=True)
    contactperson=models.CharField(max_length=100)
    designation=models.CharField(max_length=100)
    shifttopartyaddress=models.CharField(max_length=1000)
    shifttopartycity=models.CharField(max_length=100)
    shifttopartystate=models.CharField(max_length=100)
    shifttopartypostal=models.CharField(max_length=100)
    soldtopartyaddress=models.CharField(max_length=1000)
    soldtopartycity=models.CharField(max_length=100)
    soldtopartystate=models.CharField(max_length=100)
    soldtopartypostal=models.CharField(max_length=100)
    customerid = models.CharField(max_length=255, unique=True, null=True, blank=True)
    customer_type = models.CharField(max_length=100, null=True, blank=True)
    or_name = models.CharField(max_length=100, null=True, blank=True)
    or_contact = models.BigIntegerField(null=True, blank=True)


   
    def __str__(self):
        return self.fullname
   



from django.db import models
from django.utils import timezone


class Product(models.Model):
    product_id = models.AutoField(primary_key=True)
    product_name = models.CharField(max_length=255)

    CATEGORY_CHOICES = [
        ('Pest Control', 'Pest Control'),
        ('Fumigation', 'Fumigation'),
        ('Product Sale', 'Product Sale'),
    ]
   
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, default="NULL")


    def __str__(self):
        return self.product_name


    def delete_product(self):
        self.delete()


class quotation(models.Model):
    quantity = models.IntegerField()
    price = models.FloatField()
    termsandcondition = models.CharField(max_length=200)
    servicetype_q = models.CharField(max_length=5000)
    total_amount = models.FloatField(null=True, blank=True, editable=False)
    discount = models.FloatField(null=True, blank=True)
    company_name = models.CharField(max_length=100, null=True)
    company_email = models.EmailField(null=True)
    company_contact_no = models.CharField(max_length=15, default=0)
    quotation_date = models.DateField(default=timezone.now)
    company_address = models.CharField(max_length=2000, null=True)
    subject = models.CharField(max_length=200, null=True)
    total_amount_with_gst = models.FloatField(null=True, blank=True)
    gst_checkbox = models.BooleanField(default=False)
    customer = models.ForeignKey(customer_details, on_delete=models.CASCADE, null=True, blank=True)
    version = models.IntegerField(default=1)
    status = models.CharField(max_length=20, default='active')

    def save(self, *args, **kwargs):
        # Step 1: calculate total
        self.total_amount = self.quantity * self.price

        # Step 2: discount
        discounted_amount = self.total_amount
        if self.discount:
            discounted_amount -= self.total_amount * (self.discount / 100)

        # Step 3: GST handling
        if self.gst_checkbox:
            self.total_amount_with_gst = discounted_amount * 1.18
            self.gst_status = "GST"
        else:
            self.total_amount_with_gst = discounted_amount
            self.gst_status = "NON-GST"

        super().save(*args, **kwargs)

    class Meta:
        ordering = ['-version']




class invoice(models.Model):
    modeofpayment = models.CharField(max_length=100, default='Null')
    dispatchedthrough = models.CharField(max_length=100, default='Null')
    termofdelivery = models.CharField(max_length=100, default='Null')
    termsandcondition = models.CharField(max_length=255, default='Null')
    company_name = models.CharField(max_length=255, default='Null')
    company_email = models.EmailField(default='Null')
    company_contact_no = models.CharField(max_length=20, default='0')
    invoice_no = models.CharField(max_length=200, unique=True, editable=False, null=True, blank=True)
    date = models.DateField(default=timezone.now)
    description_of_goods = models.TextField(max_length=5000, default='Null')
    hsn_sac_code = models.CharField(max_length=100, default='Null')
    quantity = models.IntegerField(null=True)
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    discount = models.FloatField(null=True, blank=True)
    gst_checkbox = models.BooleanField(default=False)
    total_amount = models.FloatField(null=True, blank=True, editable=False)
    total_amount_with_gst = models.DecimalField(max_digits=10, decimal_places=2, editable=False, null=True, blank=True)
    total_amount_in_words = models.CharField(max_length=255, editable=False, default='')
    pan_card_no = models.CharField(max_length=20, default='Null')
    account_no = models.CharField(max_length=20, default='Null')
    branch = models.CharField(max_length=255, default='Null')
    ifsc_code = models.CharField(max_length=20, default='Null')
    delivery_date = models.DateField(default=timezone.now)
    dispatched_date = models.DateField(default=timezone.now)
    designation = models.CharField(max_length=20, choices=[('Indoor', 'Indoor'), ('Outdoor', 'Outdoor')], default='Null')
    customer = models.ForeignKey(customer_details, on_delete=models.CASCADE, null=True, blank=True)


    def save(self, *args, **kwargs):
        if not self.invoice_no:
            self.invoice_no = self.generate_invoice_no()


        self.total_amount = self.quantity * self.price  # Calculate total amount


        discounted_amount = self.total_amount
        if self.discount:
            discounted_amount = self.total_amount - (self.total_amount * (self.discount / 100))


        if self.gst_checkbox:
            self.total_amount_with_gst = round(discounted_amount * 1.18, 2)  # Adding 18% GST and rounding to 2 decimal places
        else:
            self.total_amount_with_gst = round(discounted_amount, 2)  # Rounding to 2 decimal places


        self.total_amount_in_words = self.convert_amount_to_words(self.total_amount_with_gst)


        super().save(*args, **kwargs)


    def save(self, *args, **kwargs):
        if self.gst_checkbox:
            self.gst_status = "GST"
        else:
            self.gst_status = "NON-GST"
        super().save(*args, **kwargs)


    def generate_invoice_no(self):
        return ''.join(random.choices('0123456789', k=11))


    def convert_amount_to_words(self, amount):
        return num2words(amount, to='currency', lang='en').replace('euro', 'rupees').replace('cents', 'paise')


    def __str__(self):
        return f"Invoice No: {self.invoice_no}"



class lead_management(models.Model):
    STATE_CHOICES = [
        ('Andhra Pradesh', 'Andhra Pradesh'),
        ('Arunachal Pradesh', 'Arunachal Pradesh'),
        ('Assam', 'Assam'),
        ('Bihar', 'Bihar'),
        ('Chhattisgarh', 'Chhattisgarh'),
        ('Goa', 'Goa'),
        ('Gujarat', 'Gujarat'),
        ('Haryana', 'Haryana'),
        ('Himachal Pradesh', 'Himachal Pradesh'),
        ('Jharkhand', 'Jharkhand'),
        ('Karnataka', 'Karnataka'),
        ('Kerala', 'Kerala'),
        ('Madhya Pradesh', 'Madhya Pradesh'),
        ('Maharashtra', 'Maharashtra'),
        ('Manipur', 'Manipur'),
        ('Meghalaya', 'Meghalaya'),
        ('Mizoram', 'Mizoram'),
        ('Nagaland', 'Nagaland'),
        ('Odisha', 'Odisha'),
        ('Punjab', 'Punjab'),
        ('Rajasthan', 'Rajasthan'),
        ('Sikkim', 'Sikkim'),
        ('Tamil Nadu', 'Tamil Nadu'),
        ('Telangana', 'Telangana'),
        ('Tripura', 'Tripura'),
        ('Uttar Pradesh', 'Uttar Pradesh'),
        ('Uttarakhand', 'Uttarakhand'),
        ('West Bengal', 'West Bengal'),
    ]

    BRANCH_CHOICES = [
        ('Bhiwandi', 'Bhiwandi'),
        ('Indore', 'Indore'),
        ('Hyderabad', 'Hyderabad'),
        ('Nagpur', 'Nagpur'),
        ('Amravti', 'Amravti'),
        ('Aurangabad', 'Aurangabad'),
        ('Baramati', 'Baramati'),
        ('Pune', 'Pune'),
    ]

    TYPEOFLEAD_CHOICES = [
        ('Hot', 'Hot'),
        ('Warm', 'Warm'),
        ('Cold', 'Cold'),
        ('Not Interested', 'Not Interested'),
        ('Loss of Order', 'Loss of Order'),
    ]

    SEGMENTS_CHOICES = [
        ('Residential', 'Residential'),
        ('Industrial / Commercial', 'Industrial / Commercial'),
        ('Institutional', 'Institutional'),
        ('Irrelevant Leads', 'Irrelevant Leads')
    ]

    state = models.CharField(max_length=100, choices=STATE_CHOICES, default="Maharashtra")
    branch = models.CharField(max_length=20, choices=BRANCH_CHOICES, default='NA')
    sourceoflead = models.CharField(max_length=200, choices=[
        ('Google', 'Google'),
        ('Justdial', 'Justdial'),
        ('Indiamart', 'Indiamart'),
        ('Customer Reference', 'Customer Reference'),
        ('BNI', 'BNI'),
        ('Lineclub', 'Lineclub'),
        ('Employee Reference', 'Employee Reference'),
        ('Others', 'Others')
    ], default="NOT SELECTED")
    salesperson = models.ForeignKey(SalesPerson, on_delete=models.CASCADE, related_name="leads")
    customername = models.CharField(max_length=100, null=True, blank=True)
    customer_type = models.CharField(max_length=100, null=True, blank=True)
    customersegment = models.CharField(max_length=100, choices=SEGMENTS_CHOICES)
    enquirydate = models.DateField(default=timezone.now)
    contactedby = models.CharField(max_length=100, null=True, blank=True)
    maincategory = models.CharField(max_length=200, null=True, blank=True)
    subcategory = models.CharField(max_length=200, null=True, blank=True)
    primarycontact = models.BigIntegerField(null=True, blank=True)
    secondarycontact = models.BigIntegerField(null=True, blank=True)
    customeremail = models.EmailField(null=True, blank=True)
    or_name = models.CharField(max_length=100, null=True, blank=True)
    or_contact = models.BigIntegerField(null=True, blank=True)
    customeraddress = models.CharField(max_length=1000, null=True, blank=True)
    location = models.URLField(null=True, blank=True)
    city = models.CharField(max_length=100, default="Unknown City")
    typeoflead = models.CharField(max_length=100, null=True, choices=TYPEOFLEAD_CHOICES)
    firstfollowupdate = models.DateField(default=timezone.now)
    stage = models.IntegerField(default=1)
    


    def __str__(self):
        return self.customername or "Unnamed Lead"

    class Meta:
        verbose_name_plural = "Lead Management"


class main_followup(models.Model):
    lead = models.ForeignKey(lead_management, on_delete=models.CASCADE)
    done_pest_control = models.CharField(max_length=10, choices=[('Yes', 'Yes'), ('No', 'No')])
    agency_name = models.CharField(max_length=255, null=True, blank=True)
    onsite_infestation = models.CharField(max_length=10, choices=[('Yes', 'Yes'), ('No', 'No')])
    infestation_level = models.CharField(max_length=10, choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')])
    typeoflead = models.CharField(max_length=100, choices=lead_management.TYPEOFLEAD_CHOICES)
    followup_remark = models.CharField(max_length=255, choices=[
        ('Call not received', 'Call not received'),
        ('Give next Follow up date', 'Give next Follow up date'),
        ('Call Out of Coverage Area', 'Call Out of Coverage Area')
    ])
    followup_comment = models.TextField()
    next_followup_date = models.DateField(null=True, blank=True)
    order_status = models.CharField(max_length=20, choices=[
        ('Close Win', 'Close Win'),
        ('Close Loss', 'Close Loss'),
        ('Not Closed', 'Not Closed')
    ], default='Not Closed')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.lead.customername or 'Unnamed'} - Followup {self.id}"






# In crmapp/models.py

class firstfollowup(models.Model):
    lead = models.ForeignKey(lead_management, on_delete=models.CASCADE, null=True, blank=True)
    havedonepestcontrolearlier = models.CharField(max_length=100, choices=[('Yes', 'Yes'),('No','No')],default='NOT SELECTED')
    agency = models.CharField(max_length=100, default="NA")
    inspectiononsite = models.CharField(max_length=100, choices=[('Yes', 'Yes'),('No','No')],default='NOT SELECTED')
    levelofinspection = models.CharField(max_length=100, choices=[('Low', 'Low'),('Middle','Middle'),('High','High')],default='NOT SELECTED')
    quotationgiven = models.CharField(max_length=100, choices=[('Yes', 'Yes'),('No','No')],default='NOT SELECTED')
    quotationamount = models.FloatField(null=True, blank=True)
    mailsent = models.CharField(max_length=100, choices=[('Yes', 'Yes'),('No','No')],default='NOT SELECTED')
    customermeeting = models.CharField(max_length=100, choices=[('Yes', 'Yes'),('No','No')],default='NOT SELECTED')
    firstremark = models.CharField(max_length=100, default="NA")
    secondfollowupdate = models.DateField(default=timezone.now)

    def __str__(self):
        return f"Follow-Up 1 for Lead {self.lead}"
    

class secondfollowup(models.Model):
    lead = models.ForeignKey(lead_management, on_delete=models.CASCADE, null=True, blank=True)
    negotiationstage = models.CharField(max_length=100, choices=[('Decision Pending', 'Decision Pending'),('Delay in Business Decision','Delay in Business Decision'),('Rates Finalized','Rates Finalized')],default='NOT SELECTED')
    mailsent2 = models.CharField(max_length=100, choices=[('Yes', 'Yes'),('No','No'),('NA','NA')],default='NOT SELECTED')
    secondremark = models.CharField(max_length=100, default="NA")
    thirdfollowupdate = models.DateField(default=timezone.now)

    def __str__(self):
        return f"Follow-Up 2 for Lead {self.lead}"
    
    
class thirdfollowup(models.Model):
    lead = models.ForeignKey(lead_management, on_delete=models.CASCADE, null=True, blank=True)
    thirdremark = models.CharField(max_length=100, default="NA")
    fourthfollowupdate = models.DateField(default=timezone.now)

    def __str__(self):
        return f"Follow-Up 3 for Lead {self.lead}"
    
class finalfollowup(models.Model):
    lead = models.ForeignKey(lead_management, on_delete=models.CASCADE, null=True, blank=True)
    fourthremark = models.CharField(max_length=100, default="NA")
    finalstatus = models.CharField(max_length=100, choices=[('Decision Pending', 'Decision Pending'),('Deal Done','Deal Done')],default='NOT SELECTED')
    contracttype = models.CharField(max_length=50, choices=[('AMC', 'AMC'), ('Single', 'Single')], default="NOT SELECTED")
    bookingamount = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Follow-Up 4 for Lead {self.lead}"



class Inventory_add(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()


    def __str__(self):
        return f'{self.product.product_name} - {self.quantity}'
   




from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator
from decimal import Decimal

class Branch(models.Model):
    branch_name = models.CharField(max_length=100)
    contact_1 = models.CharField(max_length=15)
    contact_2 = models.CharField(max_length=15, blank=True, null=True)
    email_1 = models.EmailField()
    email_2 = models.EmailField(blank=True, null=True)
    gst_number = models.CharField(max_length=20)
    pan_number = models.CharField(max_length=20)
    full_address = models.TextField()
    state = models.CharField(max_length=50 )
    code = models.IntegerField()
    shortcut = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.branch_name




class Inventory_summary(models.Model):
    customer_id = models.CharField(max_length=100, default='unknown')
    customer_name = models.CharField(max_length=255)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)

    quantity = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0.01'))]
    )

    total_price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        editable=False  
    )

    date_added = models.DateTimeField(default=timezone.now)

    @property
    def product_name(self):
        return self.product.product_name

    @property
    def price_per_unit(self):
        return self.product.price

    def save(self, *args, **kwargs):
        # Recalculate total_price accurately before saving
        self.total_price = self.quantity * self.product.price
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.customer_name} - {self.product.product_name} ({self.quantity})"


    def __str__(self):
        return f'{self.customer_details.firstname} - {self.product.product_name} - {self.total_price}'
   
class TechnicianProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    contact_number = models.CharField(max_length=15, unique=True)
    address = models.CharField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    postal_code = models.CharField(max_length=20)
    date_of_joining = models.DateField(default=timezone.now)


    def __str__(self):
        return f"{self.first_name} {self.last_name}"


class service_management(models.Model):
    SEGMENTS_CHOICES = [('Residential', 'Residential'),
        ('Industrial / Commercial', 'Industrial / Commercial'),
        ('Institutional', 'Institutional'),
        ('Irrelevant Leads', 'Irrelevant Leads')]
    customer = models.ForeignKey(customer_details, on_delete=models.CASCADE, null=True, blank=True)
    branch = models.ForeignKey("Branch", on_delete=models.CASCADE,null=True, blank=True)
    service_subject = models.CharField(max_length=500, blank=True, null=True)
    # selected_services = models.ManyToManyField(Product, related_name="selected_services")
    selected_services = models.ManyToManyField(Product, through='ServiceProduct', related_name="selected_services")
    segment = models.CharField(max_length=100, choices=SEGMENTS_CHOICES, null=True, blank=True)
    total_charges = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    total_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    total_price_with_gst = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)    
    contract_type = models.CharField(max_length=50, choices=[('One Time', 'One Time'), ('AMC', 'AMC'), ('Warranty', 'Warranty')], default="NOT SELECTED")
    contract_status = models.CharField(max_length=100, choices=[('Yes', 'Yes'), ('No', 'No')], default="NOT SELECTED")
    property_type = models.TextField(null=True, blank=True)
    warranty_period = models.CharField(max_length=50, null=True, blank=True)
    state = models.CharField(max_length=100, default="Null")
    city = models.CharField(max_length=100, default="Null")
    pincode = models.CharField(max_length=6, default="000000")
    address = models.TextField(default="Null")
    gps_location = models.URLField(null=True, blank=True)
    # gst_number = models.CharField(max_length=15, null=True, blank=True)
    frequency_count = models.CharField(max_length=50, choices=[(str(i), str(i)) for i in range(1, 13)] + [('Fortnight', 'Fortnight'), ('Weekly', 'Weekly'), ('Daily', 'Daily')], default="NOT SELECTED")
    payment_terms = models.CharField(max_length=200, default="100% Advance payment OR Whatever mutually Decided", editable=False)
    sales_person_name = models.CharField(max_length=100, null=True, blank=True)
    sales_person_contact_no = models.CharField(max_length=15, null=True, blank=True)
    delivery_time = models.TimeField(default=timezone.now)
    lead_date = models.DateField(default=timezone.now)
    service_date = models.DateField(null=True, blank=True)
    technicians = models.ManyToManyField(TechnicianProfile, blank=True, related_name='assigned_services')
    gst_status = models.CharField(
        max_length=10,
        choices=[('GST', 'GST'), ('NON-GST', 'NON-GST')],
        default='GST'
    )

    def __str__(self):
        selected_services = ', '.join([str(service) for service in self.selected_services.all()])
        return f'Service Management - {self.customer} ({selected_services})'

class ServiceProduct(models.Model):
    service = models.ForeignKey('service_management', on_delete=models.CASCADE, related_name='service_products')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.DecimalField(max_digits=10, decimal_places=2, default=1.0)
    gst_percentage = models.DecimalField(max_digits=5, decimal_places=2)
    total_with_gst = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.product.product_name} ({self.quantity} @ ₹{self.price} + GST {self.gst_percentage}%)"


from django.db import models
from django.utils import timezone
from .models import Product

from .models import QuotationTerm  # adjust path if needed
from django.db.models import Sum

# New ----------
class quotation_management(models.Model):
    customer = models.ForeignKey(customer_details, on_delete = models.CASCADE, null=True, blank=True)
    quotation_no = models.CharField(max_length=20, blank=True, null=True, unique=True)  
    contact_by = models.CharField(max_length=100 , null=True, blank=True)
    contact_by_no = models.CharField(max_length=11,null=True,blank=True)
    address = models.TextField(null=True, blank=True)
    branch = models.ForeignKey(Branch, on_delete=models.SET_NULL, null=True, blank=True)  
    selected_services = models.ManyToManyField(Product, related_name="quotation_services", blank=True)
    product_details_json = models.JSONField(null=True, blank=True)
    apply_gst = models.BooleanField(default=False)
    gst_status = models.CharField(max_length=10, default='NON-GST')
    cgst = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)  
    sgst = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)  
    igst = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)  
    gst_total = models.DecimalField(max_digits=30, decimal_places=2, null=True, blank=True)  

    total_charges = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    total_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    total_price_with_gst = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

    subject = models.CharField(max_length=1000, null=True, blank=True)
    thank_u_note = models.CharField(max_length=1000, null=True, blank=True)
    quotation_date = models.DateField(default=timezone.now)
    custom_terms = models.TextField(blank=True, null=True)
    or_name = models.CharField(max_length=100, null=True, blank=True)
    or_contact = models.CharField(max_length=10, null=True, blank=True)

    
    terms_and_conditions = models.ManyToManyField(QuotationTerm, blank=True)
    terms_order = models.JSONField(blank=True, null=True)
    gst_number = models.CharField(
        max_length=15,
        blank=True,
        null=True,
        help_text="Enter 15-digit GSTIN (optional)"
    )


    def __str__(self):
        selected_services = ', '.join([str(service) for service in self.selected_services.all()])
        customer_name = self.customer.fullname if self.customer else "No Customer"
        return f'Quotation Management - {customer_name} ({selected_services})'


    def save(self, *args, **kwargs):
        creating = self.pk is None
        super().save(*args, **kwargs)  

        if creating and not self.quotation_no:
            today = date.today()
            year = today.year
            month = str(today.month).zfill(2)
            self.quotation_no = f"{year}/{month}/{self.id}"
            super().save(update_fields=['quotation_no'])



class WorkAllocation(models.Model):
    payment_status_choice = [('Online', 'Online'), ('Cash', 'Cash'), ('Pending', 'Pending')]
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Completed', 'Completed'),
        ('workdesk','workdesk'),
    ]
    service = models.ForeignKey(service_management, on_delete=models.CASCADE, related_name='work_allocations')
    technician = models.ManyToManyField(TechnicianProfile)
    fullname = models.CharField(max_length=100)
    customer_contact = models.CharField(max_length=15) 
    customer_address = models.CharField(max_length=1000)
    gps_location = models.URLField(null=True, blank=True)
    work_description = models.TextField()
    customer_payment_status = models.CharField(max_length=20, choices=payment_status_choice)
    payment_amount = models.DecimalField(max_digits=10, decimal_places=2)
    allocated_datetime = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Pending')


    def __str__(self):
        return f"Work Allocation for {self.fullname} ({self.status})"

from django.utils.timezone import now
class Reschedule(models.Model):
    service = models.ForeignKey(service_management, on_delete=models.CASCADE, default=None, null=True, blank=True)
    old_service_date = models.DateField(default=now)
    old_delivery_time = models.TimeField(default=now)
    reason = models.TextField()

    def __str__(self):
        return f"Reschedule for Service ID {self.service.id}"   

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class TechWorkList(models.Model):
    technician = models.ForeignKey(User, on_delete=models.CASCADE)
    work = models.ManyToManyField(WorkAllocation, related_name='work')
    service = models.ForeignKey(service_management, on_delete=models.CASCADE, default=None, null=True, blank=True)
    status = models.CharField(max_length=20, choices=[('Pending', 'Pending'), ('Completed', 'Completed')], default='Pending')
    photos_before_service = models.ManyToManyField(UploadedFile, related_name='photos_before_service', blank=True)
    photos_after_service = models.ManyToManyField(UploadedFile, related_name='photos_after_service', blank=True)
    customer_signature_photo = models.ImageField(upload_to='photos/signatures/', blank=True, null=True)
    payment_photos = models.ManyToManyField(UploadedFile, related_name='payment_photos', blank=True)
    completion_datetime = models.DateTimeField(default=timezone.now)
    payment_mode = models.CharField(max_length=20, choices=[('UPI','UPI'),('Cash','Cash'),('UPI+Cash','UPI Cash')], blank=True, null=True)
    payment_type = models.CharField(max_length=20, choices=[('Full Payment','Full Payment'),('Half Payment','Half Payment')], blank=True, null=True)
    remaining_amount = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    next_due_date = models.DateField(blank=True, null=True)
    is_notified = models.BooleanField(default=False)

    
    def __str__(self):
        return f"Work by {self.technician.username}"

    


class BankAccounts(models.Model):
    bank_name = models.CharField(max_length=100)
    account_number = models.CharField(max_length=20)
    ifs_code = models.CharField(max_length=20)
    branch = models.CharField(max_length=100)

    def __str__(self):
        return self.bank_name + " - " + self.branch + " - " + self.account_number 


#  Tax Invoice model NEW
class TaxInvoice(models.Model):
    GST_TYPE_CHOICES = [
        ('CGST_SGST', 'CGST + SGST'),
        ('IGST', 'IGST'),
    ]
    quotation = models.ForeignKey(quotation_management, on_delete=models.CASCADE, related_name="tax_invoice", null=True, blank=True)
    customer = models.ForeignKey(customer_details, on_delete=models.CASCADE, related_name="customer_details")
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE, related_name="branch")
    bank = models.ForeignKey(BankAccounts, on_delete=models.CASCADE, related_name="bank_details")
    # hsn_sac = models.CharField(max_length=50, blank=True, null=True)
    tax_invoice_no = models.CharField(max_length=30, blank=True, null=True, unique=True) 
    grand_total = models.DecimalField(max_digits=10, decimal_places=2,default=Decimal('0.00')) 
    created_at = models.DateTimeField(auto_now_add=True)
    referance_no_and_date = models.CharField(max_length=100, blank=True, null=True)
    other_referance = models.CharField(max_length=500, blank=True, null=True)
    delivery_note = models.CharField(max_length=100, blank=True, null=True)
    modern_terms_of_payment = models.CharField(max_length=100, blank=True, null=True)
    buyers_order_no = models.CharField(max_length=100, blank=True, null=True)
    dated = models.DateField(blank=True, null=True)
    dispatch_doc_no = models.CharField(max_length=100, blank=True, null=True)
    delivery_note_date = models.DateField(blank=True, null=True)
    dispatched_through = models.CharField(max_length=100, blank=True, null=True)
    destination = models.CharField(max_length=1000, blank=True, null=True)
    service_titel = models.CharField(max_length=200)
    shift_gstin_uin = models.CharField(max_length=100,blank=True, null=True)
    shift_pan_number = models.CharField(max_length=50, blank=True,null=True)
    shifttopartystate = models.CharField(max_length=100)
    shifttopartystatecode = models.CharField(max_length=10)
    sold_gstin_uin = models.CharField(max_length=100, blank=True, null=True)
    sold_pan_number = models.CharField(max_length=50, blank=True,null=True)
    soldtopartystate = models.CharField(max_length=100)
    soldtopartystatecode = models.CharField(max_length=10)
    gst_type = models.CharField( max_length=20, choices=GST_TYPE_CHOICES, blank=True, null=True )
    remarks =  models.TextField(blank=True, null=True)
    terms_of_delivery = models.TextField(blank=True, null=True)
    ship_to_address = models.TextField(blank=True, null=True)
    bill_to_address = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.tax_invoice_no

    def generate_tax_invoice_no(self):
        # Step 1: Financial year
        today = date.today()
        start_year = today.year - 1 if today.month < 4 else today.year
        fy = f"{str(start_year)[-2:]}-{str(start_year + 1)[-2:]}"  

        # Step 2: Branch shortcut from quotation
        try:
            branch_code = self.quotation.branch.shortcut.upper()
        except AttributeError:
            branch_code = "NA"

        padded_id = str(self.id).zfill(5)

        return f"{fy}/{branch_code}/{padded_id}"

    def save(self, *args, **kwargs):
     creating = self.pk is None
     super().save(*args, **kwargs)  # First save to get an ID

     if creating and not self.tax_invoice_no:
         self.tax_invoice_no = self.generate_tax_invoice_no()

         # Check again to avoid duplicate tax_invoice_no (very rare case)
         if not TaxInvoice.objects.filter(tax_invoice_no=self.tax_invoice_no).exists():
             super().save(update_fields=['tax_invoice_no'])
         else:
             # In rare case of collision (e.g., deleted invoice reused ID),
             # regenerate with extra logic like appending a suffix or throw error
             suffix = str(self.id).zfill(5)
             self.tax_invoice_no = f"{self.generate_tax_invoice_no()}-{suffix}"
             super().save(update_fields=['tax_invoice_no'])


class TaxInvoiceItem(models.Model):
    tax_invoice = models.ForeignKey(TaxInvoice, on_delete=models.CASCADE, related_name='items')
    product_name = models.CharField(max_length=255)
    hsn_code = models.CharField(max_length=50, blank=True, null=True)
    quantity = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField(blank=True, null=True)
    unit = models.CharField(max_length=20, blank=True, null=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    gst_percent = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    gst_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total = models.DecimalField(max_digits=10, decimal_places=2)  

    def __str__(self):
        return f"{self.product_name} ({self.quantity})"


from django.core.exceptions import ValidationError
# Payments Record 
class PaymentsRecord(models.Model):
    PAYMENT_MODE_DICT  = {
    "UPI": "upi",
    "CASH": "cash",
    "CREDIT CARD": "credit_card",
    "DEBIT CARD": "debit_card",
    "NET BANKING": "net_banking",
    "WALLET": "wallet",
    "BANK TRANSFER": "bank_transfer",
    "CHEQUE": "cheque",
    "PAY LATER": "pay_later",
    "EMI": "emi",
    "NEFT": "neft",
    "RTGS": "rtgs",
    "IMPS": "imps"
    }

    PAYMENT_MODE_CHOICES = [(value, key) for key, value in PAYMENT_MODE_DICT.items()]

    payment_invoice_no = models.CharField(max_length=100, unique=True, blank=True)
    main_invoice = models.ForeignKey(TaxInvoice, on_delete=models.CASCADE, related_name='payments')
    amount_paid = models.DecimalField(max_digits=10, decimal_places=2)
    amount_remaining = models.DecimalField(max_digits=10, decimal_places=2, editable=False)
    payment_date = models.DateField()
    next_due_date = models.DateField(blank=True, null=True)
    previous_due_date = models.DateField(blank=True, null=True)
    work_type = models.CharField(max_length=100, blank=True, null=True)
    payment_details = models.CharField(max_length=500, blank=True, null=True)
    payment_mode = models.CharField(max_length=100, choices=PAYMENT_MODE_CHOICES)
    payment_rating = models.IntegerField(choices=[(i, f"{i} Star") for i in range(1, 6)], null=True, blank=True)
    remarks = models.TextField(blank=True, null=True)
    attachment = models.FileField(upload_to='payment_attachments/', null=True, blank=True)
    base_invoice_ref = models.CharField(max_length=100, editable=False)

    def clean(self):
        if self.attachment and self.attachment.size > 5 * 1024 * 1024:
            raise ValidationError("Attachment size cannot exceed 5MB.")
        
        creating = self.pk is None
        previous_payments = PaymentsRecord.objects.filter(main_invoice=self.main_invoice)
        if not creating:
            previous_payments = previous_payments.exclude(pk=self.pk)
    
        total_paid = previous_payments.aggregate(total=Sum('amount_paid'))['total'] or 0
        grand_total = self.main_invoice.grand_total
    
        if total_paid >= grand_total:
            raise ValidationError("This invoice is already fully paid. No further payments are allowed.")
    
        if (total_paid + self.amount_paid) > grand_total:
            raise ValidationError("Total paid exceeds invoice amount.")

    def save(self, *args, **kwargs):
        self.full_clean()  

        creating = self.pk is None

        previous_payments = PaymentsRecord.objects.filter(main_invoice=self.main_invoice)
        if not creating:
            previous_payments = previous_payments.exclude(pk=self.pk)

        total_paid = previous_payments.aggregate(total=Sum('amount_paid'))['total'] or 0
        grand_total = self.main_invoice.grand_total

        self.amount_remaining = grand_total - (total_paid + self.amount_paid)

        super().save(*args, **kwargs)

        if creating and not self.payment_invoice_no:
            base_invoice_no = self.main_invoice.tax_invoice_no or f"INV={self.main_invoice.pk}"
            count = PaymentsRecord.objects.filter(main_invoice=self.main_invoice).count()
            suffix = str(count).zfill(3)
            generated_no = f"PAY/{base_invoice_no}/{suffix}"

            if not PaymentsRecord.objects.filter(payment_invoice_no=generated_no).exists():
                self.payment_invoice_no = generated_no
                self.base_invoice_ref = base_invoice_no
            else:
                self.payment_invoice_no = f"{generated_no}-{self.pk}"

            super().save(update_fields=['payment_invoice_no','base_invoice_ref'])

    @property
    def ageing(self):
        invoice_date = None
        if hasattr(self.main_invoice, 'created_at') and self.main_invoice.created_at:
            invoice_date = self.main_invoice.created_at.date()
        elif hasattr(self.main_invoice, 'dated') and self.main_invoice.dated:
            invoice_date = self.main_invoice.dated

        if invoice_date:
            days = (timezone.now().date() - invoice_date).days
            if days <= 7:
                return "0–7 Days"
            elif days <= 15:
                return "8–15 Days"
            elif days <= 21:
                return "16–21 Days"
            elif days <= 30:
                return "22–30 Days"
            return "30+ Days"
        return "Unknown"
    
    def __str__(self):
        return self.payment_invoice_no
    

class MessageTemplates(models.Model):
    MESSAGE_TYPE_CHOICE = [
        ('email','Email'),
        ('whatsapp','WhatsApp'),
    ]

    CATEGORY_CHOICES = [
        ('lead','Lead'),
        ('service','Service Schedule'),
        ('quotation','Quotation'),
        ('invoice','Invoice'),
        ('payment','Payment Tracker'),
    ]

    LEAD_STATUS_CHOICES =[
        ('hot','Hot'),
        ('warm','Warm'),
        ('cold','Cold'),
        ('not interested','Not Interested'),
        ('loss of order','Loss of Order'),
    ]

    name = models.CharField(max_length=200, help_text="Template name (eg.Lead-hot)")
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPE_CHOICE)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    # For leads only other keep blank 
    lead_status = models.CharField(max_length=50, choices=LEAD_STATUS_CHOICES, blank=True, null=True)
    # Subject is only for email 
    subject = models.CharField(max_length=500, blank=True, null=True)
    body = models.TextField()
    attachment = models.FileField(upload_to="message_attachments/", null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.get_message_type_display()} - {self.category} - {self.name}"
    

from django.db import models
from crmapp.models import MessageTemplates, customer_details

class SentMessageLog(models.Model):

    template = models.ForeignKey(
        MessageTemplates,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )

    customer = models.ForeignKey(
        customer_details,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )

    # 🔥 NEW FIELDS (Auto populated)
    customer_name = models.CharField(max_length=255, blank=True, null=True)
    product_name = models.CharField(max_length=255, blank=True, null=True)
    recommended_product_name = models.CharField(max_length=255, blank=True, null=True)

    recipient = models.CharField(max_length=255)

    CHANNEL_CHOICES = [
        ('whatsapp', 'WhatsApp'),
        ('email', 'Email'),
    ]
    channel = models.CharField(max_length=20, choices=CHANNEL_CHOICES)

    rendered_subject = models.CharField(max_length=255, blank=True, null=True)
    rendered_body = models.TextField()
    message_id = models.CharField(max_length=100, blank=True, null=True)

    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('sent', 'Sent'),
        ('delivered', 'Delivered'),
        ('read', 'Read'),
        ('failed','Failed'),
        ('error', 'Error'),
    ]
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='queued')

    provider_response = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.channel} => {self.recipient} | {self.status}"


from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
 
# Keep the signal at the bottom
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)



class CustomerPurchase(models.Model):
    customer_id = models.IntegerField()
    product_id = models.IntegerField()
    purchase_date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Customer {self.customer_id} → Product {self.product_id}"


from django.utils import timezone


# ----------------------------------------------------------
# PURCHASE HISTORY TABLE
# ----------------------------------------------------------
from django.db import models
from crmapp.models import customer_details, Product


# ---------------- PURCHASE HISTORY MODEL ----------------
class PurchaseHistory(models.Model):
    INVOICE_TYPES = (
        ("NORMAL", "Normal Invoice"),
        ("TAX_INVOICE", "Tax Invoice"),
    )

    customer = models.ForeignKey(customer_details, on_delete=models.CASCADE)

    # For normal invoice (FK present)
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True)

    # For tax invoice (no product FK available)
    product_name = models.CharField(max_length=255, null=True, blank=True)

    quantity = models.DecimalField(max_digits=10, decimal_places=2, default=1)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)

    invoice = models.ForeignKey(
        'Invoice',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="purchase_history"
    )
    tax_invoice = models.ForeignKey(
        'TaxInvoice',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="purchase_history"
    )

    invoice_type = models.CharField(max_length=20, choices=INVOICE_TYPES, default="NORMAL")
    purchased_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        title = self.product.product_name if self.product else self.product_name
        return f"{self.customer.fullname} - {title}"
