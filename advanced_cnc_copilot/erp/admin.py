"""
Django Admin Configuration
"""

from django.contrib import admin
from .models import (
    RiseUser, Organization, Machine, Telemetry,
    Project, Tool, Job, InspectionReport,
    EconomicRecord, Alert, ConfigurationProfile
)

@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at']
    search_fields = ['name']

@admin.register(RiseUser)
class RiseUserAdmin(admin.ModelAdmin):
    list_display = ['username', 'email', 'role', 'organization']
    list_filter = ['role', 'organization']
    search_fields = ['username', 'email']

@admin.register(Machine)
class MachineAdmin(admin.ModelAdmin):
    list_display = ['name', 'controller_type', 'ip_address', 'is_active', 'organization']
    list_filter = ['controller_type', 'is_active', 'organization']
    search_fields = ['name', 'ip_address']

@admin.register(Telemetry)
class TelemetryAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'machine', 'rpm', 'load', 'signal']
    list_filter = ['machine', 'signal']
    date_hierarchy = 'timestamp'

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['part_number', 'name', 'material', 'complexity_score', 'success']
    list_filter = ['material', 'success']
    search_fields = ['part_number', 'name']

@admin.register(Tool)
class ToolAdmin(admin.ModelAdmin):
    list_display = ['tool_id', 'type', 'diameter', 'status', 'life_percentage']
    list_filter = ['type', 'status', 'material']
    search_fields = ['tool_id']
    
    def life_percentage(self, obj):
        return f"{obj.life_percentage:.1f}%"

@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ['job_id', 'project', 'machine', 'status', 'priority', 'progress_percentage']
    list_filter = ['status', 'priority']
    search_fields = ['job_id']

@admin.register(InspectionReport)
class InspectionReportAdmin(admin.ModelAdmin):
    list_display = ['part_id', 'job', 'inspector', 'passed', 'inspected_at']
    list_filter = ['passed', 'inspector']
    date_hierarchy = 'inspected_at'

@admin.register(EconomicRecord)
class EconomicRecordAdmin(admin.ModelAdmin):
    list_display = ['job', 'total_cost', 'cost_per_part', 'savings']
    search_fields = ['job__job_id']

@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ['alert_type', 'priority', 'machine', 'acknowledged', 'created_at']
    list_filter = ['alert_type', 'priority', 'acknowledged']
    date_hierarchy = 'created_at'

@admin.register(ConfigurationProfile)
class ConfigurationProfileAdmin(admin.ModelAdmin):
    list_display = ['name', 'profile_type', 'organization', 'updated_at']
    list_filter = ['profile_type', 'organization']
