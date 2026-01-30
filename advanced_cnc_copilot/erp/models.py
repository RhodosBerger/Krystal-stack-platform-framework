"""
Django Models - ERP Layer
Combines all concepts from conversation into unified data model
"""

from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
import json

# ====================
# USER MANAGEMENT
# ====================

class RiseUser(AbstractUser):
    """Extended user model with roles"""
    
    ROLE_CHOICES = [
        ('ADMIN', 'Administrator'),
        ('ENGINEER', 'Design Engineer'),
        ('OPERATOR', 'Machine Operator'),
        ('AUDITOR', 'Quality Auditor'),
    ]
    
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='OPERATOR')
    organization = models.ForeignKey('Organization', on_delete=models.CASCADE, null=True, related_name='users')
    phone = models.CharField(max_length=20, blank=True)
    
    class Meta:
        permissions = [
            ("can_override_safety", "Can override safety limits"),
            ("can_view_economics", "Can view cost analytics"),
            ("can_manage_tools", "Can manage tool inventory"),
        ]

class Organization(models.Model):
    """Multi-tenant support"""
    name = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

# ====================
# MACHINE MANAGEMENT
# ====================

class Machine(models.Model):
    """CNC Machine Registry"""
    
    CONTROLLER_CHOICES = [
        ('FANUC', 'Fanuc'),
        ('SIEMENS', 'Siemens'),
        ('HEIDENHAIN', 'Heidenhain'),
        ('HAAS', 'Haas'),
    ]
    
    name = models.CharField(max_length=100)
    controller_type = models.CharField(max_length=50, choices=CONTROLLER_CHOICES)
    ip_address = models.GenericIPAddressField()
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    is_active = models.BooleanField(default=True)
    last_heartbeat = models.DateTimeField(auto_now=True)
    
    # Specs
    axes = models.IntegerField(default=3)  # 3-axis, 5-axis, etc.
    max_rpm = models.IntegerField(default=12000)
    work_envelope = models.JSONField(default=dict)  # {"x": 500, "y": 400, "z": 300}
    
    def __str__(self):
        return f"{self.name} ({self.controller_type})"
    
    def get_current_oee(self):
        """Calculate OEE for current shift"""
        from .oee_calculator import OEECalculator
        calc = OEECalculator(self)
        return calc.get_oee()

class Telemetry(models.Model):
    """Time-series telemetry data"""
    timestamp = models.DateTimeField(db_index=True, default=timezone.now)
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE)
    
    # Core metrics
    rpm = models.IntegerField()
    load = models.FloatField()
    vibration_x = models.FloatField()
    vibration_y = models.FloatField()
    vibration_z = models.FloatField()
    
    # Temperature
    spindle_temp = models.FloatField()
    coolant_temp = models.FloatField(null=True)
    
    # Position
    position_x = models.FloatField()
    position_y = models.FloatField()
    position_z = models.FloatField()
    
    # Cognitive metrics
    dopamine = models.FloatField()
    cortisol = models.FloatField()
    signal = models.CharField(max_length=10)  # GREEN, AMBER, RED
    
    # Tool
    active_tool = models.CharField(max_length=10)
    
    class Meta:
        indexes = [
            models.Index(fields=['machine', '-timestamp']),
        ]
        ordering = ['-timestamp']

# ====================
# PROJECT MANAGEMENT
# ====================

class Project(models.Model):
    """G-code projects with AI suggestions"""
    
    name = models.CharField(max_length=200)
    part_number = models.CharField(max_length=100, unique=True)
    gcode = models.TextField()
    created_by = models.ForeignKey(RiseUser, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Material & complexity
    material = models.CharField(max_length=50)
    complexity_score = models.FloatField()
    estimated_cycle_time = models.FloatField()  # minutes
    
    # Dimensions (for similarity matching)
    dimensions = models.JSONField(default=dict)
    
    # LLM suggestions
    llm_suggestions = models.JSONField(default=dict)
    
    # Outcomes (for training)
    actual_cycle_time = models.FloatField(null=True)
    quality_score = models.FloatField(null=True)
    success = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.part_number}: {self.name}"
    
    def get_similar_projects(self, limit=5):
        """Find similar projects for LLM context"""
        # Implement similarity search
        from .similarity import find_similar
        return find_similar(self, limit)

# ====================
# TOOL MANAGEMENT
# ====================

class Tool(models.Model):
    """Tool inventory"""
    
    TOOL_TYPES = [
        ('ENDMILL', 'Endmill'),
        ('DRILL', 'Drill'),
        ('BORE', 'Boring Bar'),
        ('FACE', 'Face Mill'),
        ('THREAD', 'Thread Mill'),
    ]
    
    MATERIAL_CHOICES = [
        ('HSS', 'High Speed Steel'),
        ('HSS_CO', 'HSS-Cobalt'),
        ('CARBIDE', 'Carbide'),
        ('CERAMIC', 'Ceramic'),
        ('PCD', 'Polycrystalline Diamond'),
    ]
    
    tool_id = models.CharField(max_length=10, unique=True)  # T01, T02...
    type = models.CharField(max_length=20, choices=TOOL_TYPES)
    diameter = models.FloatField()  # mm
    length = models.FloatField()  # mm
    material = models.CharField(max_length=20, choices=MATERIAL_CHOICES)
    coating = models.CharField(max_length=50, blank=True)  # TiN, TiAlN, etc.
    
    # Lifecycle
    expected_life = models.IntegerField()  # minutes
    current_usage = models.IntegerField(default=0)  # minutes used
    purchase_price = models.DecimalField(max_digits=10, decimal_places=2)
    vendor = models.CharField(max_length=100)
    
    # Status
    STATUS_CHOICES = [
        ('NEW', 'New'),
        ('IN_USE', 'In Use'),
        ('WORN', 'Worn'),
        ('BROKEN', 'Broken'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='NEW')
    
    @property
    def remaining_life(self):
        return max(0, self.expected_life - self.current_usage)
    
    @property
    def life_percentage(self):
        return (self.remaining_life / self.expected_life) * 100
    
    def should_replace(self):
        return self.life_percentage < 10
    
    def __str__(self):
        return f"{self.tool_id}: {self.type} Ø{self.diameter}mm"

# ====================
# PRODUCTION SCHEDULING
# ====================

class Job(models.Model):
    """Production job"""
    
    job_id = models.CharField(max_length=50, unique=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    machine = models.ForeignKey(Machine, on_delete=models.SET_NULL, null=True, blank=True)
    
    quantity = models.IntegerField()
    completed_quantity = models.IntegerField(default=0)
    
    priority = models.IntegerField(default=3)  # 1=URGENT, 5=LOW
    deadline = models.DateTimeField(null=True, blank=True)
    
    STATUS_CHOICES = [
        ('QUEUED', 'Queued'),
        ('IN_PROGRESS', 'In Progress'),
        ('PAUSED', 'Paused'),
        ('COMPLETED', 'Completed'),
        ('CANCELLED', 'Cancelled'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='QUEUED')
    
    started_at = models.DateTimeField(null=True)
    completed_at = models.DateTimeField(null=True)
    
    def __str__(self):
        return f"{self.job_id}: {self.project.part_number} x{self.quantity}"
    
    @property
    def progress_percentage(self):
        return (self.completed_quantity / self.quantity) * 100

# ====================
# QUALITY INSPECTION
# ====================

class InspectionReport(models.Model):
    """Quality inspection results"""
    
    job = models.ForeignKey(Job, on_delete=models.CASCADE)
    part_id = models.CharField(max_length=100)
    inspector = models.ForeignKey(RiseUser, on_delete=models.SET_NULL, null=True)
    inspected_at = models.DateTimeField(auto_now_add=True)
    
    # Measurements
    measured_dimensions = models.JSONField()
    specified_dimensions = models.JSONField()
    
    # Result
    passed = models.BooleanField()
    non_conformances = models.JSONField(default=list)
    
    # AI analysis
    root_cause_analysis = models.TextField(blank=True)
    corrective_actions = models.TextField(blank=True)
    
    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"{self.part_id}: {status}"

# ====================
# ECONOMICS
# ====================

class EconomicRecord(models.Model):
    """Cost tracking per job"""
    
    job = models.OneToOneField(Job, on_delete=models.CASCADE)
    
    # Cost breakdown
    machine_time_cost = models.DecimalField(max_digits=10, decimal_places=2)
    material_cost = models.DecimalField(max_digits=10, decimal_places=2)
    tool_cost = models.DecimalField(max_digits=10, decimal_places=2)
    energy_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_cost = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Metrics
    cycle_time = models.FloatField()  # minutes
    cost_per_part = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Savings (vs traditional)
    traditional_cost = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    savings = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    
    def __str__(self):
        return f"{self.job.job_id}: €{self.total_cost}"

# ====================
# ALERTS & NOTIFICATIONS
# ====================

class Alert(models.Model):
    """System alerts"""
    
    ALERT_TYPES = [
        ('TOOL_REPLACEMENT', 'Tool Needs Replacement'),
        ('SAFETY', 'Safety Alert'),
        ('QUALITY', 'Quality Issue'),
        ('MAINTENANCE', 'Maintenance Required'),
        ('MATERIAL', 'Material Low Stock'),
    ]
    
    PRIORITY_CHOICES = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical'),
    ]
    
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, null=True, blank=True)
    alert_type = models.CharField(max_length=30, choices=ALERT_TYPES)
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES)
    message = models.TextField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(RiseUser, on_delete=models.SET_NULL, null=True, blank=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"[{self.priority}] {self.alert_type}: {self.message[:50]}"

# ====================
# CONFIGURATION
# ====================

class ConfigurationProfile(models.Model):
    """Dynamic form configurations"""
    
    name = models.CharField(max_length=100)
    profile_type = models.CharField(max_length=50)  # safety, dopamine, alerts, etc.
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    
    # JSON config (generated by dynamic_form_builder.py)
    config = models.JSONField()
    
    # Values
    values = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} ({self.profile_type})"
