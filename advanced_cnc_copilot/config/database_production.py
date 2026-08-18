# PostgreSQL Production Configuration
# Phase 1: Foundation - Database Migration

"""
Analogy: PostgreSQL as the City Vault
- Better security than SQLite (home safe)
- Concurrent access (multiple tellers)
- ACID compliance (audit trail)
- Advanced features (safety deposit boxes)
"""

# Install PostgreSQL
# Windows: https://www.postgresql.org/download/windows/
# Create database: cnc_copilot_prod

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'cnc_copilot_prod',
        'USER': 'cnc_admin',
        'PASSWORD': 'your_secure_password_here',
        'HOST': 'localhost',
        'PORT': '5432',
        
        # Connection Pooling (like having multiple checkout lanes)
        'CONN_MAX_AGE': 600,  # 10 minutes
        
        # Performance Tuning
        'OPTIONS': {
            'connect_timeout': 10,
            'options': '-c statement_timeout=30000',  # 30 seconds
        },
    },
    
    # Read Replica (future scaling - like branch office)
    'read_replica': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'cnc_copilot_prod',
        'USER': 'cnc_readonly',
        'PASSWORD': 'readonly_password',
        'HOST': 'replica.localhost',
        'PORT': '5432',
        'OPTIONS': {
            'options': '-c default_transaction_read_only=on'
        }
    }
}

# Database Router for Read/Write Splitting
class DatabaseRouter:
    """
    Analogy: Traffic Cop directing cars
    - Writes go to main database (one-way street)
    - Reads can use replicas (multiple lanes)
    """
    
    def db_for_read(self, model, **hints):
        """
        Reads go to replica if available (load balancing)
        """
        return 'read_replica'
    
    def db_for_write(self, model, **hints):
        """
        All writes to primary database (single source of truth)
        """
        return 'default'
    
    def allow_relation(self, obj1, obj2, **hints):
        return True
    
    def allow_migrate(self, db, app_label, model_name=None, **hints):
        return db == 'default'

# Add to settings.py
DATABASE_ROUTERS = ['path.to.DatabaseRouter']

# Logging Configuration (audit trail)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'database': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'logs/database.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django.db.backends': {
            'handlers': ['database'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}
