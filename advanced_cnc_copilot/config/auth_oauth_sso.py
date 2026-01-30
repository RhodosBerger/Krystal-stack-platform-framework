# OAuth 2.0 / SSO Authentication System
# Phase 1: Foundation - Enterprise Security

"""
Paradigm: Security as Medieval Castle Defense

OAuth Flow = Castle Entry Protocol:
1. User approaches gate (login request)
2. Guard checks credentials (authentication server)
3. Key issued (access token)
4. Token opens specific doors (resource access)
5. Token expires (limited-time pass)
6. Renewal required (refresh token)

Layers of Defense:
- Moat = Firewall
- Outer Wall = TLS/HTTPS
- Gate = OAuth/SSO
- Guards = 2FA
- Inner Keep = Protected Resources
- Patrols = Audit Logs
"""

from django.conf import settings
from allauth.socialaccount.models import SocialApp
from allauth.account.models import EmailAddress

# OAuth Providers Configuration
INSTALLED_APPS = [
    # ... existing apps
    'django.contrib.sites',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google',
    'allauth.socialaccount.providers.microsoft',
    'allauth.socialaccount.providers.github',
]

MIDDLEWARE = [
    # ... existing middleware
    'allauth.account.middleware.AccountMiddleware',
]

# Site Configuration (castle location)
SITE_ID = 1

# Authentication Backends (multiple gates)
AUTHENTICATION_BACKENDS = [
    # Local authentication (main gate)
    'django.contrib.auth.backends.ModelBackend',
    
    # OAuth/Social authentication (VIP entrance)
    'allauth.account.auth_backends.AuthenticationBackend',
]

# OAuth Settings (entry requirements)
SOCIALACCOUNT_PROVIDERS = {
    'google': {
        'SCOPE': [
            'profile',
            'email',
        ],
        'AUTH_PARAMS': {
            'access_type': 'online',
        },
        # Like VIP guest list
        'VERIFIED_EMAIL': True,
    },
    'microsoft': {
        'TENANT': 'common',  # Multi-tenant (open to all kingdoms)
        'SCOPE': [
            'User.Read',
            'email',
        ],
    },
    'github': {
        'SCOPE': [
            'user',
            'repo',
            'read:org',
        ],
    }
}

# Account Configuration (visitor policy)
ACCOUNT_AUTHENTICATION_METHOD = 'email'  # Email as passport
ACCOUNT_EMAIL_REQUIRED = True
ACCOUNT_EMAIL_VERIFICATION = 'mandatory'  # Must verify identity
ACCOUNT_USERNAME_REQUIRED = False
ACCOUNT_SIGNUP_PASSWORD_ENTER_TWICE = True  # Extra security check
ACCOUNT_SESSION_REMEMBER = True  # Remember trusted visitors

# Login/Logout URLs (castle gates)
LOGIN_URL = '/accounts/login/'
LOGIN_REDIRECT_URL = '/dashboard/'
LOGOUT_REDIRECT_URL = '/'

"""
OAuth 2.0 Flow Analogy: Airport Security

1. Authorization Request = Booking Flight
   - User selects destination (provider)
   - Provides credentials (passport)
   
2. Authorization Grant = Boarding Pass
   - Temporary code issued
   - Single-use token
   
3. Access Token = Security Badge
   - Grants access to specific areas
   - Time-limited
   - Can be revoked
   
4. Refresh Token = Frequent Flyer Card
   - Long-term credential
   - Get new access tokens
   - More secure storage required
"""

class OAuthManager:
    """Manage OAuth tokens and sessions"""
    
    @staticmethod
    def get_user_tokens(user):
        """
        Retrieve user's OAuth tokens
        Like checking visitor's credentials at checkpoint
        """
        from allauth.socialaccount.models import SocialAccount, SocialToken
        
        accounts = SocialAccount.objects.filter(user=user)
        tokens = []
        
        for account in accounts:
            try:
                token = SocialToken.objects.get(account=account)
                tokens.append({
                    'provider': account.provider,
                    'access_token': token.token,
                    'expires_at': token.expires_at,
                    'account_id': account.uid,
                })
            except SocialToken.DoesNotExist:
                continue
        
        return tokens
    
    @staticmethod
    def refresh_token_if_needed(user, provider):
        """
        Refresh expired tokens
        Like renewing visitor pass before expiry
        """
        from allauth.socialaccount.models import SocialToken
        from django.utils import timezone
        import requests
        
        try:
            token = SocialToken.objects.get(
                account__user=user,
                account__provider=provider
            )
            
            # Check if token expired (pass expired)
            if token.expires_at and token.expires_at < timezone.now():
                # Refresh logic here
                # Like going to security desk for new pass
                new_token = refresh_oauth_token(token.token_secret)
                token.token = new_token['access_token']
                token.expires_at = calculate_expiry(new_token['expires_in'])
                token.save()
                
        except SocialToken.DoesNotExist:
            return None

"""
SSO (Single Sign-On) Analogy: Hotel Key Card

Traditional Login:
- Different key for each door
- Remember multiple keys
- Lost key = locked out

SSO:
- One master key card
- Works for multiple doors
- Lost key = change one key

Benefits:
- User convenience (one password)
- Centralized control (deactivate one account)
- Better security (fewer passwords to forget)
"""

# SAML Configuration (for enterprise SSO)
SAML_ENABLED = True
SAML_IDP_METADATA_URL = 'https://login.microsoftonline.com/your-tenant/federationmetadata/2007-06/federationmetadata.xml'

SAML_CONFIG = {
    'xmlsec_binary': '/usr/bin/xmlsec1',  # Security tool
    
    # Service Provider (our castle)
    'sp': {
        'entityId': 'https://cnc-copilot.com/saml/',
        'assertionConsumerService': {
            'url': 'https://cnc-copilot.com/saml/acs/',
            'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST',
        },
        'singleLogoutService': {
            'url': 'https://cnc-copilot.com/saml/sls/',
            'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect',
        },
    },
    
    # Security settings (castle defenses)
    'security': {
        'nameIdEncrypted': True,  # Encrypt visitor names
        'authnRequestsSigned': True,  # Sign entry requests
        'wantMessagesSigned': True,  # Require signed messages
        'wantAssertionsSigned': True,  # Require signed passes
    },
}

"""
Multi-Factor Authentication (2FA) Analogy: Two-Key Safe

Traditional Login:
- One key = Knowledge (password)

2FA:
- Key 1 = Knowledge (something you know)
- Key 2 = Possession (something you have)
- Key 3 (optional) = Biometric (something you are)

Methods:
1. TOTP (Time-based One-Time Password)
   - Like rotating combination lock
   - New code every 30 seconds
   
2. SMS Code
   - Like security guard calling you
   - Code sent to phone
   
3. Hardware Token
   - Like physical key
   - YubiKey, security key
   
4. Backup Codes
   - Like spare key in lockbox
   - Use when primary methods fail
"""

# JWT Configuration (for API authentication)
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        # Session auth (cookie-based, like wristband at event)
        'rest_framework.authentication.SessionAuthentication',
        
        # JWT auth (token-based, like ticket stub)
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}

from datetime import timedelta

SIMPLE_JWT = {
    # Access token = Short-term pass (15 minutes)
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=15),
    
    # Refresh token = Long-term credential (7 days)
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    
    # Rotate refresh tokens (issue new ticket each time)
    'ROTATE_REFRESH_TOKENS': True,
    
    # Blacklist old tokens (revoke old passes)
    'BLACKLIST_AFTER_ROTATION': True,
    
    # Algorithm (encryption method)
    'ALGORITHM': 'HS256',
    
    # Signing key (master key)
    'SIGNING_KEY': settings.SECRET_KEY,
    
    # Token claims (information on pass)
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
}

"""
Permission System Analogy: Security Clearance Levels

Clearance Levels:
- Public = Marketing materials (anyone)
- Confidential = Internal docs (employees)
- Secret = Financial data (management)
- Top Secret = Source code (developers)

Django Permissions:
- IsAuthenticated = Must be employee
- IsAdminUser = Must be management
- Custom = Role-based access
"""

from rest_framework import permissions

class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Custom permission: Owner full access, others read-only
    Like: Car owner can modify, others can only look
    """
    
    def has_object_permission(self, request, view, obj):
        # Read permissions (safe methods: GET, HEAD, OPTIONS)
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions (owner only)
        return obj.owner == request.user

class IsOrganizationMember(permissions.BasePermission):
    """
    Custom permission: Organization membership required
    Like: Company badge needed for building access
    """
    
    def has_permission(self, request, view):
        return request.user.is_authenticated and \
               hasattr(request.user, 'organization')
    
    def has_object_permission(self, request, view, obj):
        return obj.organization == request.user.organization

"""
Session Management Analogy: Library Card System

Session = Library Card
- Valid for certain period
- Tracks borrowing history
- Can be suspended
- Expires if not used

Session Hijacking Protection:
- Change session ID after login (new card)
- Tie to IP address (registered address)
- Tie to user agent (signature verification)
- Short timeout (card expires)
"""

# Session Configuration
SESSION_COOKIE_SECURE = True  # HTTPS only (sealed envelope)
SESSION_COOKIE_HTTPONLY = True  # No JavaScript access (theft protection)
SESSION_COOKIE_SAMESITE = 'Lax'  # CSRF protection
SESSION_COOKIE_AGE = 86400  # 24 hours (daily renewal)

# CSRF Protection (anti-forgery)
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = True
CSRF_USE_SESSIONS = True

"""
Audit Logging Analogy: Security Camera System

Every action recorded:
- Who (user)
- What (action)
- When (timestamp)
- Where (IP addr
ess)
- How (method)
- Result (success/fail)

Like DVR:
- Continuous recording
- Searchable
- Tamper-proof
- Retention policy
"""

class AuditMiddleware:
    """
    Log all authentication events
    Like security log at building entrance
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Log request
        self.log_request(request)
        
        response = self.get_response(request)
        
        # Log response
        self.log_response(request, response)
        
        return response
    
    def log_request(self, request):
        """Record who's entering"""
        if request.user.is_authenticated:
            AuditLog.objects.create(
                user=request.user,
                action='REQUEST',
                path=request.path,
                method=request.method,
                ip_address=get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
            )
    
    def log_response(self, request, response):
        """Record what happened"""
        if response.status_code >= 400:
            # Log errors (security incidents)
            AuditLog.objects.create(
                user=request.user if request.user.is_authenticated else None,
                action='ERROR',
                path=request.path,
                status_code=response.status_code,
                ip_address=get_client_ip(request),
            )

# Helper function
def get_client_ip(request):
    """
    Get real client IP (visitor location)
    Like checking ID address
    """
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
