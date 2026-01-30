"""
Django Admin Registration for JSON Configuration Management
"""

from django.contrib import admin
from erp.json_config_models import (
    JSONConfigCategory,
    JSONConfiguration,
    JSONConfigVersion,
    JSONConfigDeployment,
    JSONConfigTemplate,
    JSONConfigAuditLog
)


@admin.register(JSONConfigCategory)
class JSONConfigCategoryAdmin(admin.ModelAdmin):
    list_display = ['category_id', 'name', 'icon', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at']


@admin.register(JSONConfiguration)
class JSONConfigurationAdmin(admin.ModelAdmin):
    list_display = ['config_id', 'name', 'category', 'version', 'status', 'updated_at']
    list_filter = ['category', 'status', 'organization']
    search_fields = ['config_id', 'name', 'description']
    readonly_fields = ['config_hash', 'created_at', 'updated_at']
    filter_horizontal = ['depends_on']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('config_id', 'name', 'category', 'description')
        }),
        ('Configuration', {
            'fields': ('config_data', 'status', 'tags')
        }),
        ('Version', {
            'fields': ('version', 'config_hash')
        }),
        ('Ownership', {
            'fields': ('organization', 'created_by', 'updated_by')
        }),
        ('Dependencies', {
            'fields': ('depends_on',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'archived_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(JSONConfigVersion)
class JSONConfigVersionAdmin(admin.ModelAdmin):
    list_display = ['config', 'version', 'changed_by', 'created_at']
    list_filter = ['config', 'created_at']
    search_fields = ['config__name', 'config__config_id', 'change_log']
    readonly_fields = ['config_hash', 'created_at']
    
    def has_add_permission(self, request):
        return False  # Versions are created automatically
    
    def has_delete_permission(self, request, obj=None):
        return False  # Cannot delete version history


@admin.register(JSONConfigDeployment)
class JSONConfigDeploymentAdmin(admin.ModelAdmin):
    list_display = ['config', 'environment', 'deployed_version', 'status', 'deployed_at', 'deployed_by']
    list_filter = ['environment', 'status', 'deployed_at']
    search_fields = ['config__name', 'config__config_id', 'deployment_notes']
    readonly_fields = ['deployed_at']


@admin.register(JSONConfigTemplate)
class JSONConfigTemplateAdmin(admin.ModelAdmin):
    list_display = ['template_id', 'name', 'category', 'is_public', 'usage_count', 'created_at']
    list_filter = ['category', 'is_public']
    search_fields = ['template_id', 'name', 'description']
    readonly_fields = ['usage_count', 'created_at']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('template_id', 'name', 'category', 'description')
        }),
        ('Template', {
            'fields': ('template_data', 'variables')
        }),
        ('Settings', {
            'fields': ('is_public', 'usage_count')
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(JSONConfigAuditLog)
class JSONConfigAuditLogAdmin(admin.ModelAdmin):
    list_display = ['config', 'action', 'performed_by', 'timestamp', 'ip_address']
    list_filter = ['action', 'timestamp', 'performed_by']
    search_fields = ['config__name', 'config__config_id', 'notes']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'
    
    def has_add_permission(self, request):
        return False  # Audit logs are created automatically
    
    def has_delete_permission(self, request, obj=None):
        return False  # Cannot delete audit logs
    
    def has_change_permission(self, request, obj=None):
        return False  # Cannot modify audit logs
