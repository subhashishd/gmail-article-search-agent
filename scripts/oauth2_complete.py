#!/usr/bin/env python3
"""
Complete OAuth2 flow for Gmail API authentication.
This script will generate a token.json file for Gmail API access.
"""

import json
import os
import urllib.parse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

def main():
    # Path to credentials
    credentials_path = './credentials/client_secret.json'
    token_path = '/tmp/token.json'  # Use writable location first
    
    # Check if credentials exist
    if not os.path.exists(credentials_path):
        print(f"Error: {credentials_path} not found!")
        return False
    
    # Load client secrets
    with open(credentials_path, 'r') as f:
        client_config = json.load(f)
    
    # Use the first configured redirect URI
    redirect_uri = 'http://localhost:8001/auth/callback'
    
    # Create flow
    flow = Flow.from_client_config(
        client_config,
        scopes=['https://www.googleapis.com/auth/gmail.readonly'],
        redirect_uri=redirect_uri
    )
    
    # Generate authorization URL
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    
    print("=" * 80)
    print("GMAIL API OAUTH2 SETUP")
    print("=" * 80)
    print()
    print("1. Open this URL in your browser:")
    print(f"   {auth_url}")
    print()
    print("2. Complete the authorization process")
    print("3. You'll be redirected to a page that says 'This site can't be reached'")
    print("4. Copy the ENTIRE URL from your browser's address bar")
    print("5. Paste it below")
    print()
    print("The URL will look like:")
    print("http://localhost:8001/auth/callback?state=...&code=...&scope=...")
    print()
    
    # Get the authorization response URL
    while True:
        auth_response_url = input("Paste the full redirect URL here: ").strip()
        
        if not auth_response_url:
            print("Please enter a URL.")
            continue
            
        if not auth_response_url.startswith('http://localhost:8001/auth/callback'):
            print("Invalid URL. Please make sure you copy the complete redirect URL.")
            continue
            
        break
    
    try:
        # Extract the authorization code from the URL
        parsed_url = urllib.parse.urlparse(auth_response_url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        if 'error' in query_params:
            print(f"Authorization error: {query_params['error'][0]}")
            return False
            
        if 'code' not in query_params:
            print("Error: No authorization code found in the URL")
            return False
            
        code = query_params['code'][0]
        print(f"✓ Authorization code extracted: {code[:20]}...")
        
        # Exchange code for tokens
        print("Exchanging code for tokens...")
        flow.fetch_token(code=code)
        
        # Save credentials
        creds = flow.credentials
        token_data = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        
        with open(token_path, 'w') as f:
            json.dump(token_data, f, indent=2)
        
        print(f"✓ Credentials saved to {token_path}")
        print("✓ OAuth2 setup complete!")
        print()
        print("You can now use the 'Fetch Articles' feature in the web interface.")
        return True
        
    except Exception as e:
        print(f"Error during token exchange: {e}")
        print("The authorization code may have expired. Please try again.")
        return False

if __name__ == "__main__":
    main()
