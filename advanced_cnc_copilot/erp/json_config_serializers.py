"""
Serializers for JSON Configuration Management
"""

from rest_framework import serializers
from .models import (
    JSONConfigCategory,
    JSONConfiguration,
    JSONConfigVersion,
    JSONConfigDeployment,
    JSONConfigTemplate,
    JSONConfigAuditLog
)


class JSONConfigCategorySerializer(serializers.ModelSerializer):
    """Serializer for configuration categories"""
    config_count = serializers.SerializerMethodField()
    
    class Meta:
        model = JSONConfigCategory
        fields = ['category_id', 'name', 'description', 'schema', 'icon', 'created_at', 'config_count']
    
    def get_config_count(self, obj):
        return obj.configs.filter(status='ACTIVE').count()


class JSONConfigVersionSerializer(serializers.ModelSerializer):
    """Serializer for configuration versions"""
    changed_by_username = serializers.CharField(source='changed_by.username', read_only=True)
    
    class Meta:
        model = JSONConfigVersion
        fields = ['id', 'version', 'config_data', 'config_hash', 'change_log', 
                  'changed_by', 'changed_by_username', 'created_at', 'diff_summary']
        read_only_fields = ['config_hash', 'created_at']


class JSONConfigurationSerializer(serializers.ModelSerializer):
    """Main serializer for JSON configurations"""
    category_name = serializers.CharField(source='category.name', read_only=True)
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    updated_by_username = serializers.CharField(source='updated_by.username', read_only=True)
    version_count = serializers.SerializerMethodField()
    
    class Meta:
        model = JSONConfiguration
        fields = [
            'id', 'config_id', 'category', 'category_name', 'name', 'description',
            'config_data', 'version', 'config_hash', 'tags', 'status',
            'organization', 'created_by', 'created_by_username',
            'updated_by', 'updated_by_username', 'created_at', 'updated_at',
            'archived_at', 'version_count'
        ]
        read_only_fields = ['config_hash', 'created_at', 'updated_at']
    
    def get_version_count(self, obj):
        return obj.version_history.count()


class JSONConfigurationDetailSerializer(JSONConfigurationSerializer):
    """Detailed serializer with version history"""
    version_history = JSONConfigVersionSerializer(many=True, read_only=True)
    depends_on = serializers.SerializerMethodField()
    dependents = serializers.SerializerMethodField()
    
    class Meta(JSONConfigurationSerializer.Meta):
        fields = JSONConfigurationSerializer.Meta.fields + ['version_history', 'depends_on', 'dependents']
    
    def get_depends_on(self, obj):
        return [{'config_id': dep.config_id, 'name': dep.name} for dep in obj.depends_on.all()]
    
    def get_dependents(self, obj):
        return [{'config_id': dep.config_id, 'name': dep.name} for dep in obj.dependents.all()]


class JSONConfigDeploymentSerializer(serializers.ModelSerializer):
    """Serializer for configuration deployments"""
    config_name = serializers.CharField(source='config.name', read_only=True)
    deployed_by_username = serializers.CharField(source='deployed_by.username', read_only=True)
    
    class Meta:
        model = JSONConfigDeployment
        fields = [
            'id', 'config', 'config_name', 'environment', 'status',
            'deployed_version', 'deployed_at', 'deployed_by', 'deployed_by_username',
            'deployment_notes', 'rollback_version'
        ]
        read_only_fields = ['deployed_at']


class JSONConfigTemplateSerializer(serializers.ModelSerializer):
    """Serializer for configuration templates"""
    category_name = serializers.CharField(source='category.name', read_only=True)
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = JSONConfigTemplate
        fields = [
            'id', 'template_id', 'category', 'category_name', 'name', 'description',
            'template_data', 'variables', 'is_public', 'usage_count',
            'created_by', 'created_by_username', 'created_at'
        ]
        read_only_fields = ['usage_count', 'created_at']


class JSONConfigAuditLogSerializer(serializers.ModelSerializer):
    """Serializer for audit logs"""
    config_name = serializers.CharField(source='config.name', read_only=True)
    performed_by_username = serializers.CharField(source='performed_by.username', read_only=True)
    
    class Meta:
        model = JSONConfigAuditLog
        fields = [
            'id', 'config', 'config_name', 'action', 'performed_by', 
            'performed_by_username', 'timestamp', 'old_value', 'new_value',
            'notes', 'ip_address', 'user_agent'
        ]
        read_only_fields = ['timestamp']
