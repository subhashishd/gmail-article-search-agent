#!/usr/bin/env python3
"""
Test script for the new bulk fetch architecture.

This script tests:
1. Bulk Gmail email fetching 
2. In-memory article extraction
3. Parallel content processing
4. Non-blocking fetch operations
"""

import requests
import json
import time
from datetime import datetime

BACKEND_URL = "http://localhost:8000"

def test_fetch_trigger():
    """Test triggering the new bulk fetch operation."""
    print("🚀 Testing bulk fetch trigger...")
    
    # Trigger fetch with a small batch for testing
    response = requests.post(f"{BACKEND_URL}/fetch")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Fetch triggered: {result}")
        return result.get("operation_id")  # Use operation_id instead of fetch_id
    else:
        print(f"❌ Fetch trigger failed: {response.status_code} - {response.text}")
        return None

def monitor_fetch_status(fetch_id):
    """Monitor the fetch status in real-time."""
    print(f"📊 Monitoring fetch status for ID: {fetch_id}")
    
    while True:
        response = requests.get(f"{BACKEND_URL}/fetch-status")
        
        if response.status_code == 200:
            status = response.json()
            
            print(f"\n--- Status Update ({datetime.now().strftime('%H:%M:%S')}) ---")
            print(f"Status: {status.get('status', 'unknown')}")
            print(f"Step: {status.get('current_step', 'unknown')}")
            print(f"Message: {status.get('message', 'No message')}")
            
            if "details" in status:
                details = status["details"]
                print(f"Progress:")
                print(f"  - Emails processed: {details.get('emails_processed', 0)}")
                print(f"  - Total emails: {details.get('total_emails', 0)}")
                print(f"  - Articles found: {details.get('articles_found', 0)}")
                print(f"  - Content fetched: {details.get('content_fetched', 0)}")
                print(f"  - Articles stored: {details.get('articles_stored', 0)}")
                print(f"  - Errors: {details.get('errors', 0)}")
            elif "progress" in status:
                print(f"Overall progress: {status['progress']}%")
            
            if "duration_seconds" in status:
                print(f"Duration: {status['duration_seconds']:.1f} seconds")
            
            # Check if completed
            status_value = status.get('status')
            if status_value in ['completed', 'failed']:
                print(f"\n🏁 Fetch {status_value}!")
                if status_value == 'completed':
                    print("✅ Bulk fetch architecture working successfully!")
                break
            
            # Continue monitoring
            time.sleep(2)
        else:
            print(f"❌ Status check failed: {response.status_code}")
            break

def test_system_stats():
    """Test system stats after fetch."""
    print("\n📈 Getting system stats...")
    
    response = requests.get(f"{BACKEND_URL}/stats/realtime")
    if response.status_code == 200:
        stats = response.json()
        print("System Stats:")
        print(json.dumps(stats, indent=2))
    else:
        print(f"❌ Stats failed: {response.status_code}")

def main():
    """Run the bulk fetch architecture test."""
    print("=" * 60)
    print("🧪 BULK FETCH ARCHITECTURE TEST")
    print("=" * 60)
    print(f"Testing new architecture:")
    print("1. Bulk Gmail fetch (fast)")
    print("2. In-memory article extraction (fast)")  
    print("3. Parallel content processing")
    print("4. Non-blocking operations")
    print()
    
    # Step 1: Trigger fetch
    fetch_id = test_fetch_trigger()
    if not fetch_id:
        print("❌ Could not start fetch, exiting")
        return
    
    # Step 2: Monitor progress
    monitor_fetch_status(fetch_id)
    
    # Step 3: Check final stats
    test_system_stats()
    
    print("\n" + "=" * 60)
    print("🎉 BULK FETCH ARCHITECTURE TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
