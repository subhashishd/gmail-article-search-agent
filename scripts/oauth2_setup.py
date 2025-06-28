#!/usr/bin/env python3
"""
OAuth2 setup script for Gmail API access.
This script runs the OAuth2 flow to generate the token.json file.
"""

import os
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Scopes required for Gmail access
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def run_oauth2_flow():
    """Run the OAuth2 authorization flow."""
    
    # Paths for credentials
    client_secrets_file = '/app/credentials/client_secret.json'
    token_file = '/app/credentials/token.json'
    
    print("Starting OAuth2 authentication flow...")
    
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
            print("Starting new OAuth2 flow...")
            
            # Check if client secrets file exists
            if not os.path.exists(client_secrets_file):
                print(f"Error: Client secrets file not found at {client_secrets_file}")
                return False
            
            # Load client secrets and check format
            with open(client_secrets_file, 'r') as f:
                client_config = json.load(f)
            
            # Handle different OAuth2 client types
            if 'web' in client_config:
                # Web application - need to convert to installed app format for local flow
                web_config = client_config['web']
                installed_config = {
                    'installed': {
                        'client_id': web_config['client_id'],
                        'client_secret': web_config['client_secret'],
                        'auth_uri': web_config['auth_uri'],
                        'token_uri': web_config['token_uri'],
                        'redirect_uris': ['http://localhost:8080']
                    }
                }
                client_config = installed_config
            
            try:
                # Create flow from client config
                flow = InstalledAppFlow.from_client_config(
                    client_config, SCOPES)
                
                # Run local server flow
                print("\nStarting local OAuth2 server...")
                print("This will open a browser window for authentication.")
                print("If the browser doesn't open automatically, copy and paste the URL shown below.")
                
                creds = flow.run_local_server(port=8080, open_browser=False)
                
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
        
        # Test searching for emails
        print("Testing email search...")
        results = service.users().messages().list(
            userId='me',
            q='from:medium.com',
            maxResults=5
        ).execute()
        
        messages = results.get('messages', [])
        print(f"✅ Found {len(messages)} emails from medium.com")
        
        if messages:
            # Get details of first message
            message = service.users().messages().get(
                userId='me',
                id=messages[0]['id'],
                format='metadata'
            ).execute()
            
            headers = {h['name']: h['value'] for h in message['payload'].get('headers', [])}
            print(f"Sample email: {headers.get('Subject', 'No subject')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gmail API: {e}")
        return False

if __name__ == "__main__":
    success = run_oauth2_flow()
    if success:
        print("\n🎉 OAuth2 setup completed successfully!")
        print("You can now use the Fetch Articles feature with your real Gmail data.")
    else:
        print("\n❌ OAuth2 setup failed. Please check the errors above.")
