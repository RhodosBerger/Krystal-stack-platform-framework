"""
URL Configuration for JSON Configuration Management API
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .json_config_views import (
    JSONConfigCategoryViewSet,
    JSONConfigurationViewSet,
    JSONConfigTemplateViewSet,
    JSONConfigAuditLogViewSet,
    JSONConfigDeploymentViewSet
)

router = DefaultRouter()
router.register(r'categories', JSONConfigCategoryViewSet, basename='json-config-category')
router.register(r'configs', JSONConfigurationViewSet, basename='json-config')
router.register(r'templates', JSONConfigTemplateViewSet, basename='json-config-template')
router.register(r'audit-logs', JSONConfigAuditLogViewSet, basename='json-config-audit')
router.register(r'deployments', JSONConfigDeploymentViewSet, basename='json-config-deployment')

urlpatterns = [
    path('', include(router.urls)),
]
