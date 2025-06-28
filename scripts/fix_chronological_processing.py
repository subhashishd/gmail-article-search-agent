#!/usr/bin/env python3
"""
Solution for Proper Chronological Email Processing

This script demonstrates the correct approach for processing Gmail emails
chronologically starting from January 1, 2025, while handling the fact that
Gmail API returns emails in reverse chronological order.

Key Components:
1. Cutoff date: January 1, 2025 (hard limit - never process emails before this)
2. Last processed date: Track the latest email we've successfully processed
3. Gmail query: Fetch emails between last_processed_date and current_date
4. Chronological sorting: Sort emails oldest-first before processing
5. Sequential processing: Process emails one by one, updating timestamp after each
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import json

class ChronologicalEmailProcessor:
    """
    Proper implementation of chronological email processing for Gmail.
    
    This class handles the complexity of Gmail's reverse chronological API
    while ensuring we process emails in the correct chronological order
    starting from January 1, 2025.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ChronologicalEmailProcessor")
        
        # Configuration
        self.CUTOFF_DATE = datetime(2025, 1, 1, 0, 0, 0)  # Hard cutoff - never process before this
        self.timestamp_file = "/app/data/last_processed_email.txt"
        self.processing_log_file = "/app/data/processing_log.json"
        
        # Processing state
        self.gmail_service = None
        self.processing_log = []
        
    def get_last_processed_date(self) -> datetime:
        """
        Get the date of the last successfully processed email.
        
        Returns:
            datetime: The last processed email date, or cutoff date if none processed yet
        """
        try:
            if os.path.exists(self.timestamp_file):
                with open(self.timestamp_file, 'r') as f:
                    timestamp_str = f.read().strip()
                    last_date = datetime.fromisoformat(timestamp_str)
                    
                    # Ensure we never go before cutoff date
                    if last_date < self.CUTOFF_DATE:
                        self.logger.warning(f"Last processed date {last_date} is before cutoff {self.CUTOFF_DATE}, using cutoff")
                        return self.CUTOFF_DATE
                    
                    return last_date
            else:
                self.logger.info(f"No timestamp file found, starting from cutoff date: {self.CUTOFF_DATE}")
                return self.CUTOFF_DATE
                
        except Exception as e:
            self.logger.error(f"Error reading timestamp file: {e}, using cutoff date")
            return self.CUTOFF_DATE
    
    def save_last_processed_date(self, email_date: datetime):
        """
        Save the date of the last successfully processed email.
        
        Args:
            email_date: The date of the email that was just processed
        """
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
                self.logger.debug(f"Email date {email_date} is not newer than last processed {current_last}, not updating")
                
        except Exception as e:
            self.logger.error(f"Error saving timestamp: {e}")
    
    def log_processing_event(self, event_type: str, data: Dict[str, Any]):
        """Log processing events for debugging and monitoring."""
        event = {
            \"timestamp\": datetime.now().isoformat(),\n            \"event_type\": event_type,\n            \"data\": data\n        }\n        \n        self.processing_log.append(event)\n        \n        # Save to file\n        try:\n            os.makedirs(os.path.dirname(self.processing_log_file), exist_ok=True)\n            with open(self.processing_log_file, 'w') as f:\n                json.dump(self.processing_log, f, indent=2, default=str)\n        except Exception as e:\n            self.logger.error(f\"Error saving processing log: {e}\")\n    \n    async def fetch_emails_for_processing(self, max_emails: int = 50) -> List[Dict[str, Any]]:\n        \"\"\"\n        Fetch emails from Gmail that need to be processed.\n        \n        This method:\n        1. Gets the last processed email date\n        2. Fetches emails from Gmail between last_processed_date and now\n        3. Filters out emails before cutoff date\n        4. Returns emails ready for chronological processing\n        \n        Args:\n            max_emails: Maximum number of emails to fetch in one batch\n            \n        Returns:\n            List of email dictionaries sorted chronologically (oldest first)\n        \"\"\"\n        try:\n            # Get date range for fetching\n            last_processed = self.get_last_processed_date()\n            current_time = datetime.now()\n            \n            self.logger.info(f\"Fetching emails from {last_processed} to {current_time}\")\n            \n            # Log fetch attempt\n            self.log_processing_event(\"fetch_attempt\", {\n                \"last_processed_date\": last_processed.isoformat(),\n                \"current_time\": current_time.isoformat(),\n                \"cutoff_date\": self.CUTOFF_DATE.isoformat(),\n                \"max_emails\": max_emails\n            })\n            \n            # Initialize Gmail service if needed\n            if self.gmail_service is None:\n                from backend.services.gmail_service_oauth import get_gmail_service\n                self.gmail_service = get_gmail_service()\n                if not self.gmail_service.authenticate():\n                    raise Exception(\"Gmail authentication failed\")\n            \n            # Fetch emails from Gmail (this returns in reverse chronological order)\n            # We query from last_processed_date to get all emails since then\n            gmail_emails = self.gmail_service.mcp_service.search_medium_emails(last_processed)\n            \n            if not gmail_emails:\n                self.logger.info(\"No new emails found\")\n                return []\n            \n            self.logger.info(f\"Fetched {len(gmail_emails)} emails from Gmail\")\n            \n            # Filter emails to ensure they're after cutoff date and after last processed\n            valid_emails = []\n            for email in gmail_emails:\n                email_date = email['date']\n                \n                # Skip emails before cutoff date (safety check)\n                if email_date < self.CUTOFF_DATE:\n                    self.logger.warning(f\"Skipping email from {email_date} (before cutoff {self.CUTOFF_DATE})\")\n                    continue\n                \n                # Skip emails we've already processed\n                if email_date <= last_processed:\n                    self.logger.debug(f\"Skipping email from {email_date} (already processed)\")\n                    continue\n                \n                valid_emails.append(email)\n            \n            # Limit number of emails if specified\n            if max_emails > 0 and len(valid_emails) > max_emails:\n                # When limiting, take the oldest emails first to maintain chronological order\n                valid_emails.sort(key=lambda x: x['date'])\n                valid_emails = valid_emails[:max_emails]\n                self.logger.info(f\"Limited to {max_emails} oldest emails for processing\")\n            \n            # Sort emails chronologically (oldest first) for processing\n            valid_emails.sort(key=lambda x: x['date'])\n            \n            if valid_emails:\n                earliest = valid_emails[0]['date']\n                latest = valid_emails[-1]['date']\n                self.logger.info(f\"Ready to process {len(valid_emails)} emails from {earliest} to {latest}\")\n            else:\n                self.logger.info(\"No new emails to process after filtering\")\n            \n            # Log successful fetch\n            self.log_processing_event(\"fetch_success\", {\n                \"total_fetched\": len(gmail_emails),\n                \"valid_emails\": len(valid_emails),\n                \"date_range\": {\n                    \"earliest\": valid_emails[0]['date'].isoformat() if valid_emails else None,\n                    \"latest\": valid_emails[-1]['date'].isoformat() if valid_emails else None\n                }\n            })\n            \n            return valid_emails\n            \n        except Exception as e:\n            self.logger.error(f\"Error fetching emails: {e}\")\n            self.log_processing_event(\"fetch_error\", {\"error\": str(e)})\n            return []\n    \n    async def process_emails_chronologically(self, max_emails: int = 20) -> Dict[str, Any]:\n        \"\"\"\n        Process emails in chronological order with proper timestamp management.\n        \n        This is the main processing method that:\n        1. Fetches emails needing processing\n        2. Processes them chronologically (oldest first)\n        3. Updates timestamp after each successful email\n        4. Handles errors gracefully without losing progress\n        \n        Args:\n            max_emails: Maximum emails to process in this batch\n            \n        Returns:\n            Dictionary with processing results\n        \"\"\"\n        start_time = datetime.now()\n        \n        try:\n            self.logger.info(f\"Starting chronological email processing (max {max_emails} emails)\")\n            \n            # Fetch emails for processing\n            emails = await self.fetch_emails_for_processing(max_emails)\n            \n            if not emails:\n                return {\n                    \"success\": True,\n                    \"message\": \"No emails to process\",\n                    \"processed_count\": 0,\n                    \"total_articles\": 0,\n                    \"processing_time\": 0\n                }\n            \n            # Process emails one by one in chronological order\n            processed_count = 0\n            total_articles = 0\n            errors = []\n            \n            for i, email in enumerate(emails, 1):\n                try:\n                    email_date = email['date']\n                    self.logger.info(f\"Processing email {i}/{len(emails)} from {email_date}: {email['subject'][:100]}...\")\n                    \n                    # Extract articles from this email\n                    articles = self.gmail_service.mcp_service.get_articles_from_email(email)\n                    \n                    if articles:\n                        # Add email metadata to articles\n                        for article in articles:\n                            article['email_date'] = email_date\n                            article['email_subject'] = email['subject']\n                            article['digest_date'] = email_date.date()\n                        \n                        # TODO: Here you would publish articles for parallel processing\n                        # await self.publish_articles_for_processing(articles)\n                        \n                        total_articles += len(articles)\n                        self.logger.info(f\"Extracted {len(articles)} articles from email\")\n                        \n                        # Log article extraction\n                        self.log_processing_event(\"articles_extracted\", {\n                            \"email_date\": email_date.isoformat(),\n                            \"email_subject\": email['subject'],\n                            \"article_count\": len(articles),\n                            \"article_titles\": [a['title'][:100] for a in articles[:5]]  # First 5 titles\n                        })\n                    \n                    # CRITICAL: Update timestamp after successful processing of each email\n                    self.save_last_processed_date(email_date)\n                    processed_count += 1\n                    \n                    # Small delay to avoid overwhelming the system\n                    await asyncio.sleep(0.1)\n                    \n                except Exception as e:\n                    error_msg = f\"Error processing email {email.get('id', 'unknown')} from {email.get('date', 'unknown')}: {e}\"\n                    self.logger.error(error_msg)\n                    errors.append(error_msg)\n                    \n                    # Log error but continue processing\n                    self.log_processing_event(\"email_processing_error\", {\n                        \"email_id\": email.get('id'),\n                        \"email_date\": email.get('date', {}).isoformat() if email.get('date') else None,\n                        \"error\": str(e)\n                    })\n                    \n                    continue\n            \n            # Calculate processing time\n            processing_time = (datetime.now() - start_time).total_seconds()\n            \n            # Final result\n            result = {\n                \"success\": True,\n                \"message\": f\"Processed {processed_count} emails with {total_articles} articles\",\n                \"processed_count\": processed_count,\n                \"total_articles\": total_articles,\n                \"processing_time\": processing_time,\n                \"errors\": errors,\n                \"last_processed_date\": self.get_last_processed_date().isoformat()\n            }\n            \n            self.logger.info(f\"Processing complete: {result['message']} in {processing_time:.2f}s\")\n            \n            # Log completion\n            self.log_processing_event(\"processing_complete\", result)\n            \n            return result\n            \n        except Exception as e:\n            error_msg = f\"Error in chronological processing: {e}\"\n            self.logger.error(error_msg)\n            \n            self.log_processing_event(\"processing_error\", {\"error\": str(e)})\n            \n            return {\n                \"success\": False,\n                \"error\": error_msg,\n                \"processed_count\": 0,\n                \"total_articles\": 0\n            }\n    \n    def get_processing_status(self) -> Dict[str, Any]:\n        \"\"\"\n        Get current processing status and statistics.\n        \n        Returns:\n            Dictionary with status information\n        \"\"\"\n        try:\n            last_processed = self.get_last_processed_date()\n            current_time = datetime.now()\n            \n            # Calculate time since last processing\n            time_since_last = current_time - last_processed\n            \n            # Get recent processing events\n            recent_events = self.processing_log[-10:] if self.processing_log else []\n            \n            return {\n                \"cutoff_date\": self.CUTOFF_DATE.isoformat(),\n                \"last_processed_date\": last_processed.isoformat(),\n                \"current_time\": current_time.isoformat(),\n                \"time_since_last_processing\": {\n                    \"total_seconds\": time_since_last.total_seconds(),\n                    \"days\": time_since_last.days,\n                    \"hours\": time_since_last.seconds // 3600\n                },\n                \"gmail_authenticated\": self.gmail_service is not None,\n                \"total_events_logged\": len(self.processing_log),\n                \"recent_events\": recent_events\n            }\n            \n        except Exception as e:\n            return {\n                \"error\": str(e),\n                \"cutoff_date\": self.CUTOFF_DATE.isoformat()\n            }\n\n\n# Example usage and testing\nasync def test_chronological_processing():\n    \"\"\"Test the chronological email processing.\"\"\"\n    processor = ChronologicalEmailProcessor()\n    \n    print(\"🧪 Testing Chronological Email Processing\")\n    print(\"=\" * 50)\n    \n    # Show current status\n    status = processor.get_processing_status()\n    print(f\"📅 Cutoff Date: {status['cutoff_date']}\")\n    print(f\"📅 Last Processed: {status['last_processed_date']}\")\n    print(f\"⏰ Time Since Last: {status['time_since_last_processing']['days']} days, {status['time_since_last_processing']['hours']} hours\")\n    \n    # Run processing\n    print(\"\\n🔄 Starting email processing...\")\n    result = await processor.process_emails_chronologically(max_emails=5)\n    \n    print(f\"\\n📊 Results:\")\n    print(f\"   Success: {result['success']}\")\n    print(f\"   Message: {result['message']}\")\n    print(f\"   Processed: {result.get('processed_count', 0)} emails\")\n    print(f\"   Articles: {result.get('total_articles', 0)}\")\n    print(f\"   Time: {result.get('processing_time', 0):.2f}s\")\n    \n    if result.get('errors'):\n        print(f\"   Errors: {len(result['errors'])}\")\n        for error in result['errors']:\n            print(f\"     - {error}\")\n    \n    # Show updated status\n    updated_status = processor.get_processing_status()\n    print(f\"\\n📅 Updated Last Processed: {updated_status['last_processed_date']}\")\n\n\nif __name__ == \"__main__\":\n    # Run the test\n    asyncio.run(test_chronological_processing())\n
