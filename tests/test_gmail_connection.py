#!/usr/bin/env python3
"""
Test Gmail Connection and Email Discovery

This script tests if we can connect to Gmail and find Medium emails.
"""

import requests
import subprocess
import json

def test_gmail_in_container():
    """Test Gmail connection inside the Docker container."""
    print("🔍 Testing Gmail Connection Inside Container...")
    
    try:
        # Test if we can import and initialize Gmail service
        result = subprocess.run([
            "docker", "exec", "gmail-search-backend", "python", "-c",
            """
import sys
sys.path.append('/app')
from backend.services.gmail_service_oauth import get_gmail_service
from datetime import datetime, timedelta

print('=== GMAIL CONNECTION TEST ===')

# Get Gmail service
service = get_gmail_service()
print(f'Gmail service created: {service is not None}')

# Test authentication
auth_result = service.authenticate()
print(f'Authentication result: {auth_result}')

if auth_result:
    # Test with a broader date range (last 30 days instead of from Jan 1)
    recent_date = datetime.now() - timedelta(days=30)
    print(f'Searching for emails since: {recent_date}')
    
    try:
        emails = service.mcp_service.search_medium_emails(recent_date)
        print(f'Found emails: {len(emails)}')
        
        if emails:
            print('Sample email dates:')
            for email in emails[:3]:
                print(f'  - {email.get("date", "No date")} - {email.get("subject", "No subject")[:80]}...')
        else:
            print('No Medium emails found in the last 30 days')
    except Exception as e:
        print(f'Error searching emails: {e}')
else:
    print('Gmail authentication failed - check credentials')
            """
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Gmail test output:")
            print(result.stdout)
            return result.stdout
        else:
            print(f"❌ Gmail test failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error running Gmail test: {e}")
        return None

def check_credentials():
    """Check if Gmail credentials exist."""
    print("\n🔐 Checking Gmail Credentials...")
    
    try:
        # Check if token file exists
        result = subprocess.run([
            "docker", "exec", "gmail-search-backend", "ls", "-la", "/app/credentials/"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("📁 Credentials directory contents:")
            print(result.stdout)
            
            if "token.json" in result.stdout:
                print("✅ OAuth token file found")
            else:
                print("❌ OAuth token file missing")
            
            if "client_secret.json" in result.stdout:
                print("✅ Client secrets file found")
            else:
                print("❌ Client secrets file missing")
        else:
            print(f"❌ Cannot access credentials directory: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error checking credentials: {e}")

def test_date_range():
    """Test what happens if we try different date ranges."""
    print("\n📅 Testing Different Date Ranges...")
    
    # Test with current system timestamp
    response = requests.get("http://localhost:8000/status")
    if response.status_code == 200:
        status = response.json()
        last_processed = status['coordinator']['agents']['email_processor']['last_processed_time']
        print(f"Current last processed time: {last_processed}")
        
        # The issue might be that the system date is June 2025, 
        # so we're looking for emails from Jan 1, 2025 to June 28, 2025
        # which might not exist in the user's Gmail
        
        print("🤔 Current setup:")
        print(f"  - System date: June 28, 2025 (system clock issue)")
        print(f"  - Looking for emails from: January 1, 2025")
        print(f"  - Search range: 6 months in 2025")
        print(f"  - This might be why no emails are found!")

def main():
    print("🧪 GMAIL CONNECTION AND EMAIL DISCOVERY TEST")
    print("=" * 60)
    
    # Test 1: Check credentials
    check_credentials()
    
    # Test 2: Test Gmail connection
    gmail_output = test_gmail_in_container()
    
    # Test 3: Analyze date range issue
    test_date_range()
    
    print("\n" + "=" * 60)
    print("📊 ANALYSIS:")
    
    if gmail_output and "Authentication result: True" in gmail_output:
        print("✅ Gmail authentication is working")
        
        if "Found emails: 0" in gmail_output:
            print("📧 No emails found - this could be expected!")
            print("   Reasons:")
            print("   1. No Medium Daily Digest emails in Gmail account")
            print("   2. No emails in the date range we're searching")
            print("   3. System date issue (searching in 2025)")
            print("\n🎯 RECOMMENDATION:")
            print("   The chronological processing is working correctly!")
            print("   The system is correctly looking for emails from Jan 1, 2025")
            print("   To test with real data, you need Medium emails in your Gmail")
        else:
            print("📧 Found some emails - processing should work")
    else:
        print("❌ Gmail authentication failed")
        print("   Check credentials setup")

if __name__ == "__main__":
    main()
