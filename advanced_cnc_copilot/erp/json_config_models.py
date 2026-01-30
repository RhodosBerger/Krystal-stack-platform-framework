"""
Django Models - JSON Configuration Management Extension
Append these models to erp/models.py
"""

from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
import json


class JSONConfigCategory(models.Model):
    """Categories for JSON configurations"""
    category_id = models.CharField(max_length=50, unique=True, primary_key=True)
    name = models.CharField(max_length=100)
    description = models.TextField()
    schema = models.JSONField(null=True, blank=True, help_text='JSON Schema for validation')
    icon = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = 'JSON Config Categories'
    
    def __str__(self):
        return self.name


class JSONConfiguration(models.Model):
    """Main JSON configuration storage with versioning"""
    
    STATUS_CHOICES = [
        ('DRAFT', 'Draft'),
        ('ACTIVE', 'Active'),
        ('ARCHIVED', 'Archived'),
        ('DEPRECATED', 'Deprecated')
    ]
    
    config_id = models.CharField(max_length=100, unique=True, db_index=True)
    category = models.ForeignKey(JSONConfigCategory, on_delete=models.CASCADE, related_name='configs')
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    # Configuration data
    config_data = models.JSONField(help_text='The actual JSON configuration')
    
    # Versioning
    version = models.CharField(max_length=20, default='1.0.0')
    config_hash = models.CharField(max_length=64, editable=False)
    
    # Metadata
    tags = models.JSONField(default=list, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='ACTIVE')
    
    # Ownership
    organization = models.ForeignKey('Organization', on_delete=models.CASCADE, related_name='json_configs')
    created_by = models.ForeignKey('RiseUser', on_delete=models.SET_NULL, null=True, related_name='created_configs')
    updated_by = models.ForeignKey('RiseUser', on_delete=models.SET_NULL, null=True, related_name='updated_configs')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    archived_at = models.DateTimeField(null=True, blank=True)
    
    # Dependencies
    depends_on = models.ManyToManyField('self', symmetrical=False, blank=True, related_name='dependents')
    
    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['config_id', 'status']),
            models.Index(fields=['category', 'status']),
            models.Index(fields=['-updated_at']),
        ]
    
    def __str__(self):
        return f"{self.name} (v{self.version})"
    
    def save(self, *args, **kwargs):
        # Generate hash for config data
        import hashlib
        config_str = json.dumps(self.config_data, sort_keys=True)
        self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        super().save(*args, **kwargs)


class JSONConfigVersion(models.Model):
    """Version history for JSON configurations"""
    config = models.ForeignKey(JSONConfiguration, on_delete=models.CASCADE, related_name='version_history')
    version = models.CharField(max_length=20)
    config_data = models.JSONField()
    config_hash = models.CharField(max_length=64)
    
    # Change tracking
    change_log = models.TextField()
    changed_by = models.ForeignKey('RiseUser', on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Diff from previous version
    diff_summary = models.JSONField(null=True, blank=True, help_text='Summary of changes from previous version')
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['config', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.config.name} v{self.version}"


class JSONConfigDeployment(models.Model):
    """Track deployments of configurations to environments"""
    
    ENVIRONMENT_CHOICES = [
        ('DEVELOPMENT', 'Development'),
        ('STAGING', 'Staging'),
        ('PRODUCTION', 'Production'),
        ('TEST', 'Test')
    ]
    
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('DEPLOYED', 'Deployed'),
        ('FAILED', 'Failed'),
        ('ROLLED_BACK', 'Rolled Back')
    ]
    
    config = models.ForeignKey(JSONConfiguration, on_delete=models.CASCADE, related_name='deployments')
    environment = models.CharField(max_length=20, choices=ENVIRONMENT_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    
    deployed_version = models.CharField(max_length=20)
    deployed_at = models.DateTimeField(auto_now_add=True)
    deployed_by = models.ForeignKey('RiseUser', on_delete=models.SET_NULL, null=True)
    
    # Deployment metadata
    deployment_notes = models.TextField(blank=True)
    rollback_version = models.CharField(max_length=20, null=True, blank=True)
    
    class Meta:
        ordering = ['-deployed_at']
        indexes = [
            models.Index(fields=['config', 'environment', '-deployed_at']),
        ]
    
    def __str__(self):
        return f"{self.config.name} → {self.environment} (v{self.deployed_version})"


class JSONConfigTemplate(models.Model):
    """Reusable templates for common configurations"""
    template_id = models.CharField(max_length=100, unique=True)
    category = models.ForeignKey(JSONConfigCategory, on_delete=models.CASCADE, related_name='templates')
    name = models.CharField(max_length=200)
    description = models.TextField()
    
    # Template structure
    template_data = models.JSONField(help_text='Template with placeholders')
    variables = models.JSONField(default=list, help_text='List of variable definitions')
    
    # Metadata
    is_public = models.BooleanField(default=False)
    usage_count = models.IntegerField(default=0)
    
    created_by = models.ForeignKey('RiseUser', on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name


class JSONConfigAuditLog(models.Model):
    """Audit log for all configuration changes"""
    
    ACTION_CHOICES = [
        ('CREATE', 'Created'),
        ('UPDATE', 'Updated'),
        ('DELETE', 'Deleted'),
        ('DEPLOY', 'Deployed'),
        ('ROLLBACK', 'Rolled Back'),
        ('ARCHIVE', 'Archived'),
        ('RESTORE', 'Restored')
    ]
    
    config = models.ForeignKey(JSONConfiguration, on_delete=models.CASCADE, related_name='audit_logs')
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    performed_by = models.ForeignKey('RiseUser', on_delete=models.SET_NULL, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Change details
    old_value = models.JSONField(null=True, blank=True)
    new_value = models.JSONField(null=True, blank=True)
    notes = models.TextField(blank=True)
    
    # Request metadata
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=500, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['config', '-timestamp']),
            models.Index(fields=['performed_by', '-timestamp']),
        ]
    
    def __str__(self):
        return f"{self.action} {self.config.name} by {self.performed_by} at {self.timestamp}"
