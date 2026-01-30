"""
Set admin password
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fanuc_rise_django.settings')
django.setup()

from erp.models import RiseUser

try:
    user = RiseUser.objects.get(username='admin')
    user.set_password('admin123')  
    user.save()
    print(f"✅ Password set for user: {user.username}")
    print(f"   Login: admin")
    print(f"   Password: admin123")
except RiseUser.DoesNotExist:
    print("❌ Admin user not found")
