# Redis Cache Configuration
# Phase 1: Foundation - Caching Layer

"""
Paradigm: Redis as the Bloodstream
- Fast transport of frequently accessed data
- Reduces load on database (heart)
- Expires old data automatically (metabolism)
- Pub/Sub for real-time communication (nervous system)
"""

# Redis Configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None  # Set in production

# Django Cache Configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'PASSWORD': REDIS_PASSWORD,
            
            # Connection Pool (like having reserve blood)
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            },
            
            # Serialization (like oxygen binding to hemoglobin)
            'SERIALIZER': 'django_redis.serializers.json.JSONSerializer',
            
            # Compression (like compressing gases in bloodstream)
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
        },
        'KEY_PREFIX': 'cnc',
        'TIMEOUT': 300,  # 5 minutes default
    },
    
    # Session Cache (short-term memory)
    'sessions': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': f'redis://{REDIS_HOST}:{REDIS_PORT}/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'KEY_PREFIX': 'session',
        'TIMEOUT': 3600,  # 1 hour
    },
    
    # API Response Cache (quick reflexes)
    'api': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': f'redis://{REDIS_HOST}:{REDIS_PORT}/2',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'KEY_PREFIX': 'api',
        'TIMEOUT': 60,  # 1 minute
    },
}

# Session Configuration (using Redis)
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'sessions'

# Example: Cache Decorator Usage
from django.views.decorators.cache import cache_page
from django.core.cache import cache

"""
Analogy: Cache Strategies as Water Management

1. Write-Through (Direct Pipe):
   - Write to cache and database simultaneously
   - Like filling both storage tank and direct pipe
   
2. Write-Back (Delayed Write):
   - Write to cache first, database later
   - Like filling storage tank, then draining to reservoir
   
3. Cache-Aside (Lazy Loading):
   - Check cache, if miss, load from database
   - Like checking fridge before going to store
"""

class CacheStrategies:
    """Cache pattern implementations"""
    
    @staticmethod
    def write_through(key, value, timeout=300):
        """
        Pattern: Direct Pipe
        - Immediate consistency
        - Slower writes
        - Best for: Critical data
        """
        cache.set(key, value, timeout)
        # Also save to database immediately
        return value
    
    @staticmethod
    def cache_aside(key, fetch_function, timeout=300):
        """
        Pattern: Lazy Loading (most common)
        - Check cache first
        - Load from DB on miss
        - Best for: Read-heavy workloads
        """
        value = cache.get(key)
        if value is None:
            value = fetch_function()
            cache.set(key, value, timeout)
        return value
    
    @staticmethod
    def write_back(key, value, timeout=300):
        """
        Pattern: Delayed Write
        - Fast writes
        - Eventual consistency
        - Best for: High-write scenarios
        """
        cache.set(key, value, timeout)
        # Database write happens asynchronously (Celery task)
        return value

# Cache Key Patterns (like filing system)
class CacheKeys:
    """
    Analogy: Library Dewey Decimal System
    - Organized, predictable keys
    - Easy to invalidate related data
    """
    
    # Machine data (like Science section 500-599)
    MACHINE = 'machine:{machine_id}'
    MACHINE_TELEMETRY = 'machine:{machine_id}:telemetry'
    MACHINE_STATUS = 'machine:{machine_id}:status'
    
    # Job data (like Technology section 600-699)
    JOB = 'job:{job_id}'
    JOB_STATUS = 'job:{job_id}:status'
    
    # User data (like Arts section 700-799)
    USER_PROFILE = 'user:{user_id}:profile'
    USER_PERMISSIONS = 'user:{user_id}:permissions'
    
    # Dashboard (like Reference section)
    DASHBOARD = 'dashboard:{dashboard_id}'
    DASHBOARD_CONFIG = 'dashboard:{dashboard_id}:config'
    
    @staticmethod
    def invalidate_machine(machine_id):
        """
        Invalidate all cache entries for a machine
        Like clearing entire shelf in library
        """
        pattern = f'machine:{machine_id}:*'
        cache.delete_pattern(pattern)

# Cache Warming (pre-loading frequently accessed data)
def warm_cache():
    """
    Analogy: Pre-heating oven
    - Load hot data into cache before users request it
    - Improves first-request performance
    """
    from erp.models import Machine
    
    # Load all active machines
    machines = Machine.objects.filter(is_active=True)
    
    for machine in machines:
        key = CacheKeys.MACHINE.format(machine_id=machine.id)
        cache.set(key, machine, timeout=3600)

# Rate Limiting using Redis (traffic cop)
from django.core.cache import cache
from django.http import HttpResponse

def rate_limit(max_requests=100, window=60):
    """
    Analogy: Turnstile at event
    - Only allow X people per minute
    - Prevents stampede
    """
    def decorator(view_func):
        def wrapper(request, *args, **kwargs):
            ip = request.META.get('REMOTE_ADDR')
            key = f'rate_limit:{ip}'
            
            # Get current count
            count = cache.get(key, 0)
            
            if count >= max_requests:
                return HttpResponse('Rate limit exceeded', status=429)
            
            # Increment counter
            cache.set(key, count + 1, window)
            
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator

# Example Usage
@cache_page(60, cache='api')  # Cache for 1 minute
@rate_limit(max_requests=100, window=60)  # 100 requests per minute
def get_machine_status(request, machine_id):
    """
    Cached API endpoint with rate limiting
    Like express lane at grocery store (fast, but limited)
    """
    key = CacheKeys.MACHINE_STATUS.format(machine_id=machine_id)
    
    status = CacheStrategies.cache_aside(
        key=key,
        fetch_function=lambda: get_status_from_database(machine_id),
        timeout=30  # 30 seconds
    )
    
    return JsonResponse(status)

"""
Cache Eviction Strategies (like cleaning refrigerator)

1. LRU (Least Recently Used):
   - Remove items not accessed recently
   - Like throwing out food you haven't eaten
   
2. LFU (Least Frequently Used):
   - Remove items used least often
   - Like getting rid of that weird ingredient
   
3. FIFO (First In, First Out):
   - Remove oldest items first
   - Like rotating stock at grocery store
   
4. TTL (Time To Live):
   - Automatic expiration
   - Like milk expiration date
"""

# Redis for WebSocket (Pub/Sub)
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [(REDIS_HOST, REDIS_PORT)],
            'capacity': 1500,  # Max messages in channel
            'expiry': 10,  # Message expiry in seconds
        },
    },
}

"""
Pub/Sub Analogy: Radio Broadcasting
- Publisher = Radio Station
- Subscriber = Radio Receiver
- Channel = Frequency
- Message = Broadcast
"""

# Monitoring Cache Performance
class CacheMetrics:
    """
    Track cache effectiveness
    Like measuring blood pressure
    """
    
    @staticmethod
    def get_hit_rate():
        """
        Hit rate = successful cache retrievals / total requests
        Like batting average in baseball
        """
        info = cache.get_client().info()
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        
        if hits + misses == 0:
            return 0
        
        return hits / (hits + misses)
    
    @staticmethod
    def get_memory_usage():
        """
        Memory usage tracking
        Like checking fuel gauge
        """
        info = cache.get_client().info('memory')
        used_memory = info.get('used_memory_human', '0B')
        max_memory = info.get('maxmemory_human', '0B')
        
        return {
            'used': used_memory,
            'max': max_memory,
        }
