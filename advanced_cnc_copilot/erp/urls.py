"""
ERP URLs Configuration
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'machines', views.MachineViewSet)
router.register(r'projects', views.ProjectViewSet)
router.register(r'tools', views.ToolViewSet)
router.register(r'jobs', views.JobViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('config/<str:config_type>/', views.get_form_config, name='get-config'),
    path('config/<str:config_type>/save/', views.save_form_config, name='save-config'),
    path('analytics/dashboard/', views.get_dashboard_analytics, name='dashboard-analytics'),
    
    # JSON Configuration Management API
    path('json-configs/', include('erp.json_config_urls')),
]
