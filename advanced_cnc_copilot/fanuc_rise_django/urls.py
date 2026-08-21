"""
Django URL Configuration
"""

from django.contrib import admin
from django.urls import path, include
from .views import home_view

urlpatterns = [
    path('', home_view, name='home'),  # Homepage
    path('admin/', admin.site.urls),
    path('api/', include('erp.urls')),
    path('api-auth/', include('rest_framework.urls')),
]
