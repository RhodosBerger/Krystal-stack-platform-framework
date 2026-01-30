"""
API Views for JSON Configuration Management
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.utils import timezone
import hashlib
import json

from .models import (
    JSONConfigCategory,
    JSONConfiguration,
    JSONConfigVersion,
    JSONConfigDeployment,
    JSONConfigTemplate,
    JSONConfigAuditLog
)
from .json_config_serializers import (
    JSONConfigCategorySerializer,
    JSONConfigurationSerializer,
    JSONConfigurationDetailSerializer,
    JSONConfigVersionSerializer,
    JSONConfigDeploymentSerializer,
    JSONConfigTemplateSerializer,
    JSONConfigAuditLogSerializer
)
from .json_config_manager import JSONConfigRegistry


class JSONConfigCategoryViewSet(viewsets.ModelViewSet):
    """ViewSet for JSON configuration categories"""
    queryset = JSONConfigCategory.objects.all()
    serializer_class = JSONConfigCategorySerializer
    lookup_field = 'category_id'
    
    @action(detail=True, methods=['get'])
    def configs(self, request, category_id=None):
        """Get all configs in a category"""
        category = self.get_object()
        configs = category.configs.filter(status='ACTIVE')
        serializer = JSONConfigurationSerializer(configs, many=True)
        return Response(serializer.data)


class JSONConfigurationViewSet(viewsets.ModelViewSet):
    """ViewSet for JSON configurations with version management"""
    queryset = JSONConfiguration.objects.all()
    lookup_field = 'config_id'
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return JSONConfigurationDetailSerializer
        return JSONConfigurationSerializer
    
    def get_queryset(self):
        queryset = JSONConfiguration.objects.all()
        
        # Filter by category
        category = self.request.query_params.get('category')
        if category:
            queryset = queryset.filter(category_id=category)
        
        # Filter by status
        status_filter = self.request.query_params.get('status', 'ACTIVE')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Filter by tags
        tags = self.request.query_params.get('tags')
        if tags:
            tag_list = tags.split(',')
            for tag in tag_list:
                queryset = queryset.filter(tags__contains=[tag.strip()])
        
        # Search
        search = self.request.query_params.get('search')
        if search:
            queryset = queryset.filter(
                models.Q(name__icontains=search) |
                models.Q(description__icontains=search) |
                models.Q(config_id__icontains=search)
            )
        
        return queryset
    
    def perform_create(self, serializer):
        """Create configuration with audit log"""
        config = serializer.save(
            created_by=self.request.user,
            updated_by=self.request.user
        )
        
        # Create initial version
        JSONConfigVersion.objects.create(
            config=config,
            version=config.version,
            config_data=config.config_data,
            config_hash=config.config_hash,
            change_log='Initial version',
            changed_by=self.request.user
        )
        
        # Create audit log
        self._create_audit_log(config, 'CREATE')
    
    def perform_update(self, serializer):
        """Update configuration with versioning"""
        old_instance = self.get_object()
        old_data = old_instance.config_data
        old_version = old_instance.version
        
        # Save updated instance
        config = serializer.save(updated_by=self.request.user)
        
        # Check if config_data changed
        if old_data != config.config_data:
            # Bump version
            new_version = self._bump_version(old_version)
            config.version = new_version
            config.save()
            
            # Create version history entry
            JSONConfigVersion.objects.create(
                config=config,
                version=new_version,
                config_data=config.config_data,
                config_hash=config.config_hash,
                change_log=self.request.data.get('change_log', f'Updated to v{new_version}'),
                changed_by=self.request.user
            )
            
            # Create audit log
            self._create_audit_log(config, 'UPDATE', old_value=old_data, new_value=config.config_data)
    
    @action(detail=True, methods=['post'])
    def archive(self, request, config_id=None):
        """Archive a configuration"""
        config = self.get_object()
        config.status = 'ARCHIVED'
        config.archived_at = timezone.now()
        config.save()
        
        self._create_audit_log(config, 'ARCHIVE')
        
        return Response({'status': 'archived'})
    
    @action(detail=True, methods=['post'])
    def restore(self, request, config_id=None):
        """Restore an archived configuration"""
        config = self.get_object()
        config.status = 'ACTIVE'
        config.archived_at = None
        config.save()
        
        self._create_audit_log(config, 'RESTORE')
        
        return Response({'status': 'restored'})
    
    @action(detail=True, methods=['post'])
    def rollback(self, request, config_id=None):
        """Rollback to a previous version"""
        config = self.get_object()
        target_version = request.data.get('version')
        
        if not target_version:
            return Response({'error': 'version is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Find the target version
        try:
            version_obj = config.version_history.get(version=target_version)
        except JSONConfigVersion.DoesNotExist:
            return Response({'error': f'Version {target_version} not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Store old data for audit
        old_data = config.config_data
        old_version = config.version
        
        # Rollback
        config.config_data = version_obj.config_data
        config.version = self._bump_version(config.version)
        config.save()
        
        # Create version entry
        JSONConfigVersion.objects.create(
            config=config,
            version=config.version,
            config_data=config.config_data,
            config_hash=config.config_hash,
            change_log=f'Rolled back to v{target_version}',
            changed_by=request.user
        )
        
        self._create_audit_log(config, 'ROLLBACK', notes=f'Rolled back from v{old_version} to v{target_version}')
        
        return Response({
            'status': 'rolled_back',
            'old_version': old_version,
            'target_version': target_version,
            'new_version': config.version
        })
    
    @action(detail=True, methods=['get'])
    def validate(self, request, config_id=None):
        """Validate configuration against schema"""
        config = self.get_object()
        
        if not config.category.schema:
            return Response({'valid': True, 'message': 'No schema defined for validation'})
        
        # Use JSONConfigRegistry for validation
        registry = JSONConfigRegistry()
        is_valid, error = registry.validate_config(config.config_data, config.category.category_id)
        
        return Response({
            'valid': is_valid,
            'error': error
        })
    
    @action(detail=True, methods=['post'])
    def deploy(self, request, config_id=None):
        """Deploy configuration to an environment"""
        config = self.get_object()
        environment = request.data.get('environment')
        notes = request.data.get('notes', '')
        
        if not environment:
            return Response({'error': 'environment is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Create deployment record
        deployment = JSONConfigDeployment.objects.create(
            config=config,
            environment=environment,
            status='DEPLOYED',
            deployed_version=config.version,
            deployed_by=request.user,
            deployment_notes=notes
        )
        
        self._create_audit_log(config, 'DEPLOY', notes=f'Deployed to {environment}')
        
        serializer = JSONConfigDeploymentSerializer(deployment)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def export(self, request, config_id=None):
        """Export configuration as JSON file"""
        config = self.get_object()
        
        export_data = {
            'metadata': {
                'config_id': config.config_id,
                'name': config.name,
                'category': config.category.category_id,
                'version': config.version,
                'exported_at': timezone.now().isoformat()
            },
            'config': config.config_data
        }
        
        return Response(export_data)
    
    @action(detail=False, methods=['post'])
    def import_config(self, request):
        """Import configuration from JSON"""
        import_data = request.data
        
        if 'metadata' not in import_data or 'config' not in import_data:
            return Response({'error': 'Invalid import format'}, status=status.HTTP_400_BAD_REQUEST)
        
        metadata = import_data['metadata']
        config_data = import_data['config']
        
        # Get or create category
        category, _ = JSONConfigCategory.objects.get_or_create(
            category_id=metadata['category'],
            defaults={'name': metadata['category'].title(), 'description': ''}
        )
        
        # Create configuration
        config = JSONConfiguration.objects.create(
            config_id=metadata['config_id'],
            name=metadata.get('name', metadata['config_id']),
            category=category,
            config_data=config_data,
            organization=request.user.organization,
            created_by=request.user,
            updated_by=request.user
        )
        
        self._create_audit_log(config, 'CREATE', notes='Imported configuration')
        
        serializer = JSONConfigurationSerializer(config)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    def _bump_version(self, current_version: str, bump_type: str = 'patch') -> str:
        """Bump semantic version"""
        try:
            major, minor, patch = map(int, current_version.split('.'))
        except:
            return '1.0.0'
        
        if bump_type == 'major':
            major += 1
            minor = 0
            patch = 0
        elif bump_type == 'minor':
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        return f"{major}.{minor}.{patch}"
    
    def _create_audit_log(self, config, action, old_value=None, new_value=None, notes=''):
        """Create audit log entry"""
        JSONConfigAuditLog.objects.create(
            config=config,
            action=action,
            performed_by=self.request.user,
            old_value=old_value,
            new_value=new_value,
            notes=notes,
            ip_address=self._get_client_ip(),
            user_agent=self.request.META.get('HTTP_USER_AGENT', '')[:500]
        )
    
    def _get_client_ip(self):
        """Get client IP address"""
        x_forwarded_for = self.request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = self.request.META.get('REMOTE_ADDR')
        return ip


class JSONConfigTemplateViewSet(viewsets.ModelViewSet):
    """ViewSet for configuration templates"""
    queryset = JSONConfigTemplate.objects.all()
    serializer_class = JSONConfigTemplateSerializer
    lookup_field = 'template_id'
    
    def get_queryset(self):
        queryset = JSONConfigTemplate.objects.all()
        
        # Show public templates + user's private templates
        if not self.request.user.is_staff:
            queryset = queryset.filter(
                models.Q(is_public=True) | models.Q(created_by=self.request.user)
            )
        
        # Filter by category
        category = self.request.query_params.get('category')
        if category:
            queryset = queryset.filter(category_id=category)
        
        return queryset
    
    @action(detail=True, methods=['post'])
    def instantiate(self, request, template_id=None):
        """Create a new configuration from template"""
        template = self.get_object()
        
        # Get variable values from request
        variable_values = request.data.get('variables', {})
        
        # Replace placeholders in template
        config_data = self._replace_placeholders(template.template_data, variable_values)
        
        # Create configuration
        config = JSONConfiguration.objects.create(
            config_id=request.data.get('config_id'),
            name=request.data.get('name'),
            category=template.category,
            config_data=config_data,
            organization=request.user.organization,
            created_by=request.user,
            updated_by=request.user
        )
        
        # Increment usage count
        template.usage_count += 1
        template.save()
        
        serializer = JSONConfigurationSerializer(config)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    def _replace_placeholders(self, template_data, variables):
        """Replace {{variable}} placeholders with actual values"""
        import re
        
        template_str = json.dumps(template_data)
        
        for var_name, var_value in variables.items():
            placeholder = f"{{{{{var_name}}}}}"
            template_str = template_str.replace(placeholder, str(var_value))
        
        return json.loads(template_str)


class JSONConfigAuditLogViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for audit logs (read-only)"""
    queryset = JSONConfigAuditLog.objects.all()
    serializer_class = JSONConfigAuditLogSerializer
    
    def get_queryset(self):
        queryset = JSONConfigAuditLog.objects.all()
        
        # Filter by config
        config_id = self.request.query_params.get('config_id')
        if config_id:
            queryset = queryset.filter(config__config_id=config_id)
        
        # Filter by action
        action = self.request.query_params.get('action')
        if action:
            queryset = queryset.filter(action=action)
        
        # Filter by user
        user_id = self.request.query_params.get('user_id')
        if user_id:
            queryset = queryset.filter(performed_by_id=user_id)
        
        return queryset


class JSONConfigDeploymentViewSet(viewsets.ModelViewSet):
    """ViewSet for configuration deployments"""
    queryset = JSONConfigDeployment.objects.all()
    serializer_class = JSONConfigDeploymentSerializer
    
    def get_queryset(self):
        queryset = JSONConfigDeployment.objects.all()
        
        # Filter by environment
        environment = self.request.query_params.get('environment')
        if environment:
            queryset = queryset.filter(environment=environment)
        
        # Filter by config
        config_id = self.request.query_params.get('config_id')
        if config_id:
            queryset = queryset.filter(config__config_id=config_id)
        
        return queryset
    
    @action(detail=True, methods=['post'])
    def rollback_deployment(self, request, pk=None):
        """Rollback a deployment"""
        deployment = self.get_object()
        
        if not deployment.rollback_version:
            return Response({'error': 'No rollback version specified'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Create new deployment with rollback version
        new_deployment = JSONConfigDeployment.objects.create(
            config=deployment.config,
            environment=deployment.environment,
            status='DEPLOYED',
            deployed_version=deployment.rollback_version,
            deployed_by=request.user,
            deployment_notes=f'Rollback from v{deployment.deployed_version}'
        )
        
        # Mark old deployment as rolled back
        deployment.status = 'ROLLED_BACK'
        deployment.save()
        
        serializer = self.get_serializer(new_deployment)
        return Response(serializer.data)
