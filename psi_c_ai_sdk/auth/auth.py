"""
Authentication Layer for ΨC-AI SDK

This module provides authentication mechanisms for the ΨC-AI SDK,
including API key validation and integration points for external
authentication systems.
"""

import time
import logging
import os
import hashlib
import hmac
import json
import uuid
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from functools import wraps
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ApiKey:
    """API key with metadata for authentication."""
    
    key: str
    name: str = "default"
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if the API key is still valid (not expired)."""
        if self.expires_at is None:
            return True
        return time.time() < self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if the API key has a specific permission."""
        # If permissions list is empty, allow all permissions
        if not self.permissions:
            return True
        return permission in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApiKey':
        """Create an ApiKey from a dictionary."""
        return cls(**data)


class AuthManager:
    """
    Authentication manager for the ΨC-AI SDK.
    
    Handles API key validation, generation, and management,
    with support for future integration with external auth systems.
    """
    
    def __init__(
        self,
        auth_file: Optional[str] = None,
        secret_key: Optional[str] = None,
        require_auth: bool = True
    ):
        """
        Initialize the authentication manager.
        
        Args:
            auth_file: Path to authentication file (default: None)
            secret_key: Secret key for signing tokens (default: auto-generated)
            require_auth: Whether authentication is required (default: True)
        """
        self.auth_file = auth_file
        self.secret_key = secret_key or os.environ.get("PSI_SECRET_KEY") or self._generate_secret_key()
        self.require_auth = require_auth
        
        # Store API keys in memory
        self.api_keys: Dict[str, ApiKey] = {}
        
        # Track failed authentication attempts
        self.failed_attempts: Dict[str, List[float]] = {}
        
        # Load API keys if auth file exists
        if auth_file and os.path.exists(auth_file):
            self.load_keys()
    
    def _generate_secret_key(self) -> str:
        """Generate a random secret key."""
        return str(uuid.uuid4())
    
    def generate_api_key(
        self,
        name: str = "default",
        expires_in: Optional[float] = None,
        permissions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ApiKey:
        """
        Generate a new API key.
        
        Args:
            name: Name for the API key
            expires_in: Optional seconds until expiration
            permissions: Optional list of permissions
            metadata: Optional metadata for the key
            
        Returns:
            The generated API key
        """
        # Generate a random key with format: prefix.random
        key_value = f"psk_{uuid.uuid4().hex}"
        
        # Calculate expiration time if provided
        expires_at = None
        if expires_in is not None:
            expires_at = time.time() + expires_in
            
        # Create the API key
        api_key = ApiKey(
            key=key_value,
            name=name,
            created_at=time.time(),
            expires_at=expires_at,
            permissions=permissions or [],
            metadata=metadata or {}
        )
        
        # Store the key
        self.api_keys[key_value] = api_key
        
        # Save keys to file if specified
        if self.auth_file:
            self.save_keys()
            
        return api_key
    
    def validate_key(self, api_key: str) -> Tuple[bool, Optional[str]]:
        """
        Validate an API key.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check if authentication is required
        if not self.require_auth:
            return True, None
            
        # Check if key exists
        if api_key not in self.api_keys:
            return False, "Invalid API key"
            
        # Get the key
        key_obj = self.api_keys[api_key]
        
        # Check if key is expired
        if not key_obj.is_valid():
            return False, "Expired API key"
            
        return True, None
    
    def validate_permission(self, api_key: str, permission: str) -> Tuple[bool, Optional[str]]:
        """
        Validate an API key has a specific permission.
        
        Args:
            api_key: The API key to validate
            permission: The permission to check
            
        Returns:
            Tuple of (has_permission, reason_if_not)
        """
        # First validate the key itself
        valid, reason = self.validate_key(api_key)
        if not valid:
            return valid, reason
            
        # Check permission
        key_obj = self.api_keys[api_key]
        if not key_obj.has_permission(permission):
            return False, f"Missing permission: {permission}"
            
        return True, None
    
    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: The API key to revoke
            
        Returns:
            True if key was revoked, False if not found
        """
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            
            # Save keys to file if specified
            if self.auth_file:
                self.save_keys()
                
            return True
        return False
    
    def load_keys(self) -> None:
        """Load API keys from file."""
        try:
            with open(self.auth_file, 'r') as f:
                data = json.load(f)
            
            self.api_keys = {}
            for key_data in data.get('api_keys', []):
                api_key = ApiKey.from_dict(key_data)
                self.api_keys[api_key.key] = api_key
                
            logger.info(f"Loaded {len(self.api_keys)} API keys from {self.auth_file}")
                
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
    
    def save_keys(self) -> None:
        """Save API keys to file."""
        try:
            # Convert API keys to dictionaries
            keys_data = [key.to_dict() for key in self.api_keys.values()]
            
            # Create data structure
            data = {
                'api_keys': keys_data
            }
            
            # Save to file
            with open(self.auth_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(self.api_keys)} API keys to {self.auth_file}")
                
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
    
    def log_failed_attempt(self, client_id: str) -> None:
        """
        Log a failed authentication attempt.
        
        Args:
            client_id: Identifier for the client (e.g., IP address)
        """
        if client_id not in self.failed_attempts:
            self.failed_attempts[client_id] = []
            
        self.failed_attempts[client_id].append(time.time())
        
        # Clean up old attempts (older than 1 hour)
        cutoff = time.time() - 3600
        self.failed_attempts[client_id] = [
            attempt for attempt in self.failed_attempts[client_id]
            if attempt >= cutoff
        ]
    
    def is_rate_limited(self, client_id: str, max_attempts: int = 5) -> bool:
        """
        Check if a client is rate limited due to too many failed attempts.
        
        Args:
            client_id: Identifier for the client (e.g., IP address)
            max_attempts: Maximum allowed attempts in the past hour
            
        Returns:
            True if client is rate limited, False otherwise
        """
        if client_id not in self.failed_attempts:
            return False
            
        # Clean up old attempts (older than 1 hour)
        cutoff = time.time() - 3600
        self.failed_attempts[client_id] = [
            attempt for attempt in self.failed_attempts[client_id]
            if attempt >= cutoff
        ]
        
        # Check if too many attempts
        return len(self.failed_attempts[client_id]) >= max_attempts

    def get_all_keys(self) -> List[ApiKey]:
        """Get all API keys."""
        return list(self.api_keys.values())


# Helper functions for web frameworks

def create_auth_middleware(auth_manager: AuthManager):
    """
    Create an authentication middleware for web frameworks.
    
    This is a factory function that can be used to create 
    middleware for different web frameworks. The specific
    implementation would depend on the framework being used.
    
    Example factory implementation for FastAPI:
    
    ```python
    def create_fastapi_auth_middleware(auth_manager: AuthManager):
        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Get API key from header or query parameter
            api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
            
            # Get client IP for rate limiting
            client_ip = request.client.host
            
            # Check if rate limited
            if auth_manager.is_rate_limited(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many failed attempts, please try again later"}
                )
            
            # Validate API key
            valid, reason = auth_manager.validate_key(api_key)
            if not valid:
                auth_manager.log_failed_attempt(client_ip)
                return JSONResponse(
                    status_code=401,
                    content={"detail": reason or "Unauthorized"}
                )
            
            # Continue processing the request
            return await call_next(request)
    ```
    """
    pass


def requires_api_key(auth_manager: AuthManager, permission: Optional[str] = None):
    """
    Decorator for requiring API key authentication.
    
    This is a generic implementation that would need to be
    adapted for specific frameworks. The implementation shown
    here is for illustration purposes.
    
    For actual usage in a framework like FastAPI:
    
    ```python
    auth_manager = AuthManager(auth_file="keys.json")
    
    @app.get("/protected")
    @requires_api_key(auth_manager, permission="read:data")
    def protected_route():
        return {"message": "This route is protected"}
    ```
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get API key from arguments or context
            # This would be framework-specific
            api_key = "..."  # Extract from context
            client_id = "..."  # Extract from context
            
            # Check if rate limited
            if auth_manager.is_rate_limited(client_id):
                # Return rate limit error
                # This would be framework-specific
                return {"error": "Rate limited"}
            
            # Validate API key and permission
            if permission:
                valid, reason = auth_manager.validate_permission(api_key, permission)
            else:
                valid, reason = auth_manager.validate_key(api_key)
                
            if not valid:
                auth_manager.log_failed_attempt(client_id)
                # Return authentication error
                # This would be framework-specific
                return {"error": reason or "Unauthorized"}
            
            # Call the original function
            return func(*args, **kwargs)
        return wrapper
    return decorator 