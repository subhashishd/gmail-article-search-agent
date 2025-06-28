"""OAuth2-based Gmail API service for fetching and processing Medium articles."""

import os
import hashlib
from typing import List, Dict
from datetime import datetime

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

class OAuth2GmailService:
    """Service for interacting with Gmail API using OAuth2 user authentication."""

    def __init__(self):
        self.service = None
        self.scopes = ['https://www.googleapis.com/auth/gmail.readonly']
        self.token_file = '/app/credentials/token.json'
        self.client_secrets_file = '/app/credentials/client_secret.json'

    def authenticate(self) -> bool:
        """Authenticate with Gmail API using OAuth2 flow."""
        try:
            creds = None
            # Check if token file exists
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
            
            # If there are no (valid) credentials available, return False for manual setup
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    # Save the credentials for the next run
                    with open(self.token_file, 'w') as token:
                        token.write(creds.to_json())
                else:
                    print("❌ Gmail credentials not found or invalid. Please run the OAuth setup.")
                    return False
            
            self.service = build('gmail', 'v1', credentials=creds)
            print("✅ Gmail API OAuth2 authentication successful")
            return True
            
        except Exception as e:
            print(f"❌ Gmail authentication failed: {e}")
            return False

    def search_medium_emails(self, last_update: datetime) -> List[Dict]:
        """Search for Medium emails since the last update, with pagination."""
        if not self.service and not self.authenticate():
            return []

        try:
            date_query = f"after:{last_update.strftime('%Y/%m/%d')}"
            query = f"from:(noreply@medium.com) {date_query}"
            
            messages = []
            page_token = None

            while True:
                results = self.service.users().messages().list(
                    userId='me', q=query, pageToken=page_token).execute()
                
                messages.extend(results.get('messages', []))
                page_token = results.get('nextPageToken')
                
                if not page_token:
                    break

            print(f"Found {len(messages)} Medium emails since {last_update}")

            email_details = []
            for msg in messages:
                email_content = self.get_email_content(msg['id'])
                if email_content:
                    email_details.append(email_content)
            return email_details

        except Exception as e:
            print(f"Error searching emails: {e}")
            return []

    def get_email_content(self, message_id: str) -> Dict:
        """Get email content by message ID."""
        try:
            message = self.service.users().messages().get(
                userId='me', id=message_id, format='full').execute()

            headers = {h['name']: h['value'] for h in message['payload'].get('headers', [])}
            body = self._extract_email_body(message['payload'])
            date_str = headers.get('Date', '')
            email_date = datetime.now()
            if date_str:
                from email.utils import parsedate_to_datetime
                email_date = parsedate_to_datetime(date_str)


            return {
                'id': message_id,
                'subject': headers.get('Subject', ''),
                'sender': headers.get('From', ''),
                'date': email_date,
                'body': body
            }

        except Exception as e:
            print(f"Error getting email content: {e}")
            return {}

    def _extract_email_body(self, payload) -> str:
        """Extract email body from message payload."""
        body = ""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/html':
                    if 'data' in part['body']:
                        import base64
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
        elif payload['mimeType'] == 'text/html' and 'data' in payload['body']:
            import base64
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        return body

    def get_articles_from_email(self, email_content: Dict) -> List[Dict]:
        """Extract Medium article links and titles from email content."""
        articles = []
        try:
            soup = BeautifulSoup(email_content.get('body', ''), 'html.parser')
            links = soup.find_all('a', href=True)

            for link in links:
                href = link['href']
                if 'medium.com' in href and '-' in href:
                    title = link.get_text(strip=True)
                    if title and len(title) > 20:
                        article_hash = hashlib.sha256((title + href).encode('utf-8')).hexdigest()
                        articles.append({
                            'title': title,
                            'link': href,
                            'summary': f"Summary for: {title}", # Placeholder summary
                            'hash': article_hash,
                            'processed_at': datetime.now().isoformat(),
                            'digest_date': email_content['date'].date()
                        })
            return articles
        except Exception as e:
            print(f"Error extracting articles from email: {e}")
            return []

class GmailServiceOAuth2:
    """Wrapper class for OAuth2 Gmail service."""

    def __init__(self):
        self.mcp_service = OAuth2GmailService()

    def authenticate(self) -> bool:
        return self.mcp_service.authenticate()

# gmail_service_oauth2 = GmailServiceOAuth2()  # Disabled for manual initialization

# Lazy initialization function
_gmail_service_instance = None

def get_gmail_service():
    """Get Gmail service instance with lazy initialization."""
    global _gmail_service_instance
    if _gmail_service_instance is None:
        _gmail_service_instance = GmailServiceOAuth2()
    return _gmail_service_instance
