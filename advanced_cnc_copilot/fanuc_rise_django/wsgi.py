"""
WSGI config for fanuc_rise_django project.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fanuc_rise_django.settings')

application = get_wsgi_application()
