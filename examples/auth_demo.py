#!/usr/bin/env python3
"""
Auth Layer Demo

This example demonstrates how to use the authentication layer
for API key management and validation.
"""

import os
import time
from psi_c_ai_sdk.auth import AuthManager, ApiKey


def main():
    print("ΨC-AI SDK: Auth Layer Demo")
    print("=" * 40)
    
    # Create an auth manager with a temporary file
    auth_file = "demo_keys.json"
    auth_manager = AuthManager(auth_file=auth_file)
    
    print("\n1. Generating API keys")
    print("-" * 40)
    
    # Generate a standard API key
    standard_key = auth_manager.generate_api_key(
        name="Standard Key",
        permissions=[]  # Empty list = all permissions
    )
    print(f"Standard Key: {standard_key.key}")
    print(f"Created at: {time.ctime(standard_key.created_at)}")
    print(f"Expires: {'Never' if standard_key.expires_at is None else time.ctime(standard_key.expires_at)}")
    print(f"Permissions: {'All (empty list)' if not standard_key.permissions else ', '.join(standard_key.permissions)}")
    
    # Generate a key with limited permissions
    limited_key = auth_manager.generate_api_key(
        name="Limited Key",
        permissions=["read:memories", "read:schema"],
        expires_in=3600  # Expires in 1 hour
    )
    print(f"\nLimited Key: {limited_key.key}")
    print(f"Created at: {time.ctime(limited_key.created_at)}")
    print(f"Expires: {time.ctime(limited_key.expires_at)}")
    print(f"Permissions: {', '.join(limited_key.permissions)}")
    
    # Save keys to file
    auth_manager.save_keys()
    print(f"\nSaved keys to {auth_file}")
    
    print("\n2. Validating API keys")
    print("-" * 40)
    
    # Validate the standard key
    valid, reason = auth_manager.validate_key(standard_key.key)
    print(f"Standard key valid: {valid}")
    
    # Validate permissions
    valid, reason = auth_manager.validate_permission(standard_key.key, "write:memories")
    print(f"Standard key has 'write:memories' permission: {valid}")
    
    valid, reason = auth_manager.validate_permission(limited_key.key, "write:memories")
    print(f"Limited key has 'write:memories' permission: {valid}")
    print(f"Reason: {reason}")
    
    valid, reason = auth_manager.validate_permission(limited_key.key, "read:memories")
    print(f"Limited key has 'read:memories' permission: {valid}")
    
    print("\n3. Managing API keys")
    print("-" * 40)
    
    # Get all keys
    all_keys = auth_manager.get_all_keys()
    print(f"Number of keys: {len(all_keys)}")
    
    # Revoke a key
    revoked = auth_manager.revoke_key(limited_key.key)
    print(f"Revoked limited key: {revoked}")
    
    # Check key count after revocation
    all_keys = auth_manager.get_all_keys()
    print(f"Number of keys after revocation: {len(all_keys)}")
    
    # Try to validate the revoked key
    valid, reason = auth_manager.validate_key(limited_key.key)
    print(f"Revoked key valid: {valid}")
    print(f"Reason: {reason}")
    
    print("\n4. Loading keys from file")
    print("-" * 40)
    
    # Create a new auth manager that loads from the file
    new_auth_manager = AuthManager(auth_file=auth_file)
    all_keys = new_auth_manager.get_all_keys()
    print(f"Loaded {len(all_keys)} keys from file")
    
    if all_keys:
        key = all_keys[0]
        print(f"Key name: {key.name}")
        print(f"Key value: {key.key}")
    
    # Clean up the demo file
    try:
        os.remove(auth_file)
        print(f"\nCleaned up demo file: {auth_file}")
    except:
        pass
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main() 