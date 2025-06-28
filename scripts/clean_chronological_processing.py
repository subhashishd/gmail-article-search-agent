#!/usr/bin/env python3
"""
Clean Solution for Proper Chronological Email Processing

This demonstrates the correct approach for Gmail chronological processing:
1. Cutoff date: January 1, 2025 (never process emails before this)
2. Last processed date: Track latest successfully processed email
3. Fetch emails in reverse chronological order from Gmail API
4. Sort and process emails chronologically (oldest first)
5. Update timestamp after each email to prevent reprocessing
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import List, Dict, Any
import json

class CleanChronologicalProcessor:
    """Clean implementation of chronological email processing for Gmail."""
    
    def __init__(self):
        self.logger = logging.getLogger("ChronologicalProcessor")
        
        # Configuration
        self.CUTOFF_DATE = datetime(2025, 1, 1, 0, 0, 0)  # Hard cutoff date
        self.timestamp_file = "/app/data/last_processed_email.txt"
        
        # State
        self.gmail_service = None
        
    def get_last_processed_date(self) -> datetime:
        """Get the date of the last successfully processed email."""
        try:
            if os.path.exists(self.timestamp_file):
                with open(self.timestamp_file, 'r') as f:
                    timestamp_str = f.read().strip()
                    last_date = datetime.fromisoformat(timestamp_str)
                    
                    # Ensure we never go before cutoff date
                    if last_date < self.CUTOFF_DATE:
                        self.logger.warning(f"Last processed date {last_date} is before cutoff, using cutoff")
                        return self.CUTOFF_DATE
                    
                    return last_date
            else:
                self.logger.info(f"No timestamp file found, starting from cutoff date: {self.CUTOFF_DATE}")
                return self.CUTOFF_DATE
                
        except Exception as e:
            self.logger.error(f"Error reading timestamp file: {e}, using cutoff date")
            return self.CUTOFF_DATE
    
    def save_last_processed_date(self, email_date: datetime):
        """Save the date of the last successfully processed email."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.timestamp_file), exist_ok=True)
            
            # Only update if this email is newer than our last processed date
            current_last = self.get_last_processed_date()
            if email_date > current_last:
                with open(self.timestamp_file, 'w') as f:
                    f.write(email_date.isoformat())
                
                self.logger.info(f"Updated last processed date to: {email_date}")
            else:
                self.logger.debug(f"Email date {email_date} is not newer than last processed {current_last}")
                
        except Exception as e:
            self.logger.error(f"Error saving timestamp: {e}")
    
    async def fetch_emails_for_processing(self, max_emails: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch emails from Gmail that need to be processed.
        
        Returns emails sorted chronologically (oldest first) for processing.
        """
        try:
            # Get date range for fetching
            last_processed = self.get_last_processed_date()
            current_time = datetime.now()
            
            self.logger.info(f"Fetching emails from {last_processed} to {current_time}")
            
            # Initialize Gmail service if needed
            if self.gmail_service is None:
                from backend.services.gmail_service_oauth import get_gmail_service
                self.gmail_service = get_gmail_service()
                if not self.gmail_service.authenticate():
                    raise Exception("Gmail authentication failed")
            
            # Fetch emails from Gmail (returns in reverse chronological order)
            gmail_emails = self.gmail_service.mcp_service.search_medium_emails(last_processed)
            
            if not gmail_emails:
                self.logger.info("No new emails found")
                return []
            
            self.logger.info(f"Fetched {len(gmail_emails)} emails from Gmail")
            
            # Filter emails to ensure they're after cutoff date and after last processed
            valid_emails = []
            for email in gmail_emails:
                email_date = email['date']
                
                # Skip emails before cutoff date (safety check)
                if email_date < self.CUTOFF_DATE:
                    self.logger.warning(f"Skipping email from {email_date} (before cutoff)")
                    continue
                
                # Skip emails we've already processed
                if email_date <= last_processed:
                    self.logger.debug(f"Skipping email from {email_date} (already processed)")
                    continue
                
                valid_emails.append(email)
            
            # Limit number of emails if specified
            if max_emails > 0 and len(valid_emails) > max_emails:
                # When limiting, take the oldest emails first to maintain chronological order
                valid_emails.sort(key=lambda x: x['date'])
                valid_emails = valid_emails[:max_emails]
                self.logger.info(f"Limited to {max_emails} oldest emails for processing")
            
            # Sort emails chronologically (oldest first) for processing
            valid_emails.sort(key=lambda x: x['date'])
            
            if valid_emails:
                earliest = valid_emails[0]['date']
                latest = valid_emails[-1]['date']
                self.logger.info(f"Ready to process {len(valid_emails)} emails from {earliest} to {latest}")
            else:
                self.logger.info("No new emails to process after filtering")
            
            return valid_emails
            
        except Exception as e:
            self.logger.error(f"Error fetching emails: {e}")
            return []
    
    async def process_emails_chronologically(self, max_emails: int = 20) -> Dict[str, Any]:
        """
        Process emails in chronological order with proper timestamp management.
        
        This is the main processing method that:
        1. Fetches emails needing processing
        2. Processes them chronologically (oldest first)
        3. Updates timestamp after each successful email
        4. Handles errors gracefully without losing progress
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting chronological email processing (max {max_emails} emails)")
            
            # Fetch emails for processing
            emails = await self.fetch_emails_for_processing(max_emails)
            
            if not emails:
                return {
                    "success": True,
                    "message": "No emails to process",
                    "processed_count": 0,
                    "total_articles": 0,
                    "processing_time": 0
                }
            
            # Process emails one by one in chronological order
            processed_count = 0
            total_articles = 0
            errors = []
            
            for i, email in enumerate(emails, 1):
                try:
                    email_date = email['date']
                    email_subject = email.get('subject', 'No Subject')
                    self.logger.info(f"Processing email {i}/{len(emails)} from {email_date}: {email_subject[:100]}...")
                    
                    # Extract articles from this email
                    articles = self.gmail_service.mcp_service.get_articles_from_email(email)
                    
                    if articles:
                        # Add email metadata to articles
                        for article in articles:
                            article['email_date'] = email_date
                            article['email_subject'] = email_subject
                            article['digest_date'] = email_date.date()
                        
                        # TODO: Here you would publish articles for parallel processing
                        # This is where the event-driven architecture would take over
                        # await self.publish_articles_for_processing(articles)
                        
                        total_articles += len(articles)
                        self.logger.info(f"Extracted {len(articles)} articles from email")
                    
                    # CRITICAL: Update timestamp after successful processing of each email
                    self.save_last_processed_date(email_date)
                    processed_count += 1
                    
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    error_msg = f"Error processing email {email.get('id', 'unknown')} from {email.get('date', 'unknown')}: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                    
                    # Continue processing other emails even if one fails
                    continue
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Final result
            result = {
                "success": True,
                "message": f"Processed {processed_count} emails with {total_articles} articles",
                "processed_count": processed_count,
                "total_articles": total_articles,
                "processing_time": processing_time,
                "errors": errors,
                "last_processed_date": self.get_last_processed_date().isoformat()
            }
            
            self.logger.info(f"Processing complete: {result['message']} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Error in chronological processing: {e}"
            self.logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "processed_count": 0,
                "total_articles": 0
            }
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status and statistics."""
        try:
            last_processed = self.get_last_processed_date()
            current_time = datetime.now()
            
            # Calculate time since last processing
            time_since_last = current_time - last_processed
            
            return {
                "cutoff_date": self.CUTOFF_DATE.isoformat(),
                "last_processed_date": last_processed.isoformat(),
                "current_time": current_time.isoformat(),
                "time_since_last_processing": {
                    "total_seconds": time_since_last.total_seconds(),
                    "days": time_since_last.days,
                    "hours": time_since_last.seconds // 3600
                },
                "gmail_authenticated": self.gmail_service is not None
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "cutoff_date": self.CUTOFF_DATE.isoformat()
            }


# Test and demonstration
async def main():
    """Test the chronological email processing."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    processor = CleanChronologicalProcessor()
    
    print("🧪 Testing Clean Chronological Email Processing")
    print("=" * 60)
    
    # Show current status
    status = processor.get_processing_status()
    print(f"📅 Cutoff Date: {status['cutoff_date']}")
    print(f"📅 Last Processed: {status['last_processed_date']}")
    print(f"⏰ Time Since Last: {status['time_since_last_processing']['days']} days, {status['time_since_last_processing']['hours']} hours")
    
    # Run processing
    print("\n🔄 Starting email processing...")
    result = await processor.process_emails_chronologically(max_emails=5)
    
    print(f"\n📊 Results:")
    print(f"   ✅ Success: {result['success']}")
    print(f"   📧 Message: {result['message']}")
    print(f"   📊 Processed: {result.get('processed_count', 0)} emails")
    print(f"   📄 Articles: {result.get('total_articles', 0)}")
    print(f"   ⏱️  Time: {result.get('processing_time', 0):.2f}s")
    
    if result.get('errors'):
        print(f"   ❌ Errors: {len(result['errors'])}")
        for error in result['errors'][:3]:  # Show first 3 errors
            print(f"     - {error}")
    
    # Show updated status
    updated_status = processor.get_processing_status()
    print(f"\n📅 Updated Last Processed: {updated_status['last_processed_date']}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
