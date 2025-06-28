#!/usr/bin/env python3
"""
Manual OAuth2 setup script for Gmail API access.
This script runs a manual OAuth2 flow that doesn't require a local server.
"""

import os
import json
import urllib.parse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# Scopes required for Gmail access
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def run_manual_oauth2_flow():
    """Run the manual OAuth2 authorization flow."""
    
    # Paths for credentials
    client_secrets_file = '/app/credentials/client_secret.json'
    token_file = '/app/credentials/token.json'
    
    print("Starting Manual OAuth2 authentication flow...")
    
    # Check if we already have valid credentials
    creds = None
    if os.path.exists(token_file):
        print("Found existing token file. Checking validity...")
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired token...")
            try:
                creds.refresh(Request())
                print("Token refreshed successfully!")
            except Exception as e:
                print(f"Token refresh failed: {e}")
                creds = None
        
        if not creds:
            print("Starting new manual OAuth2 flow...")
            
            # Check if client secrets file exists
            if not os.path.exists(client_secrets_file):
                print(f"Error: Client secrets file not found at {client_secrets_file}")
                return False
            
            # Load client secrets and check format
            with open(client_secrets_file, 'r') as f:
                client_config = json.load(f)
            
            # Handle different OAuth2 client types
            if 'web' in client_config:
                # Use the first redirect URI from the web config
                redirect_uri = client_config['web']['redirect_uris'][0]
                client_config_for_flow = {
                    'web': {
                        'client_id': client_config['web']['client_id'],
                        'client_secret': client_config['web']['client_secret'],
                        'auth_uri': client_config['web']['auth_uri'],
                        'token_uri': client_config['web']['token_uri'],
                        'redirect_uris': [redirect_uri]
                    }
                }
            elif 'installed' in client_config:
                # Use installed app config
                redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'  # Out-of-band flow
                client_config_for_flow = client_config
            else:
                print("Unknown client configuration format")
                return False
            
            try:
                # Create flow from client config
                flow = Flow.from_client_config(
                    client_config_for_flow, 
                    scopes=SCOPES,
                    redirect_uri=redirect_uri
                )
                
                # Generate authorization URL
                auth_url, _ = flow.authorization_url(
                    access_type='offline',
                    include_granted_scopes='true'
                )
                
                print("\n" + "="*70)
                print("MANUAL OAUTH2 AUTHORIZATION")
                print("="*70)
                print("\n1. Copy and paste this URL into your browser:")
                print(f"\n{auth_url}\n")
                print("2. Complete the authorization in your browser")
                print("3. After authorization, you'll be redirected to a page")
                print("4. Copy the 'code' parameter from the URL or the authorization code shown")
                print("5. Paste it below when prompted")
                print("\n" + "="*70)
                
                # Get authorization code from user
                auth_code = input("\nPaste the authorization code here: ").strip()
                
                if not auth_code:
                    print("No authorization code provided. Aborting.")
                    return False
                
                # Exchange authorization code for tokens
                print("Exchanging authorization code for tokens...")
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                
                print("OAuth2 flow completed successfully!")
                
            except Exception as e:
                print(f"OAuth2 flow failed: {e}")
                return False
        
        # Save the credentials for the next run
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
        print(f"Token saved to {token_file}")
    
    # Test the credentials by accessing Gmail API
    try:
        print("Testing Gmail API access...")
        service = build('gmail', 'v1', credentials=creds)
        profile = service.users().getProfile(userId='me').execute()
        print(f"✅ Successfully authenticated Gmail for: {profile.get('emailAddress')}")
        
        # Test searching for Medium emails
        print("Testing Medium email search...")
        
        # Try different search queries for Medium
        search_queries = [
            'from:noreply@medium.com',
            'from:medium.com',
            'medium.com',
            'subject:"Daily Digest"'
        ]
        
        total_found = 0
        for query in search_queries:
            try:
                results = service.users().messages().list(
                    userId='me',
                    q=query,
                    maxResults=10
                ).execute()
                
                messages = results.get('messages', [])
                if messages:
                    print(f"✅ Query '{query}': Found {len(messages)} emails")
                    total_found += len(messages)
                    
                    # Get details of first message
                    if total_found == len(messages):  # Only for first successful query
                        message = service.users().messages().get(
                            userId='me',
                            id=messages[0]['id'],
                            format='metadata'
                        ).execute()
                        
                        headers = {h['name']: h['value'] for h in message['payload'].get('headers', [])}
                        print(f"   Sample email: {headers.get('Subject', 'No subject')}")
                        print(f"   From: {headers.get('From', 'Unknown sender')}")
                else:
                    print(f"❌ Query '{query}': No emails found")
                    
            except Exception as e:
                print(f"❌ Query '{query}' failed: {e}")
        
        if total_found > 0:
            print(f"\n🎉 Total Medium-related emails found: {total_found}")
        else:
            print(f"\n⚠️  No Medium emails found. You may need to:")
            print("   - Subscribe to Medium Daily Digest")
            print("   - Check if you have Medium emails in your Gmail")
            print("   - Try the Fetch Articles feature anyway (it will use mock data)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gmail API: {e}")
        return False

if __name__ == "__main__":
    success = run_manual_oauth2_flow()
    if success:
        print("\n🎉 OAuth2 setup completed successfully!")
        print("You can now use the Fetch Articles feature with your real Gmail data.")
    else:
        print("\n❌ OAuth2 setup failed. Please check the errors above.")
