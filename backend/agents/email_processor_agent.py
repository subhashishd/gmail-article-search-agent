"""
Chronological Email Processing Agent - Clean Implementation

This agent implements proper chronological email processing:
1. Hard cutoff date: January 1, 2025 (never process emails before this)
2. Chronological processing: Always process emails oldest-first
3. Proper timestamp management: Update after each successful email
4. Crash recovery: Resume from last successfully processed email
5. Event-driven integration: Publishes articles for parallel processing
"""

import asyncio
import logging
import os
from typing import Dict, Any, List
from datetime import datetime, timezone
from backend.core.event_bus import event_bus
from backend.services.gmail_service_oauth import get_gmail_service
from backend.core.memory_manager import memory_manager
from backend.core.task_planner import HTNPlanner, MCTSPlanner, TaskExecutor
from backend.core.tool_registry import ToolRegistry

class EmailProcessorAgent:
    """Chronological email processing agent with proper timestamp management"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"EmailProcessorAgent-{self.agent_id}")
        
        # Configuration - Hard cutoff date (never process emails before this)
        self.CUTOFF_DATE = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.timestamp_file = "/app/data/last_processed_email.txt"
        
        # State
        self.gmail_service = None
        
    async def initialize(self):
        """Initialize Email Processor Agent"""
        self.logger.info("Initializing Chronological Email Processor Agent...")
        self.logger.info(f"Hard cutoff date: {self.CUTOFF_DATE}")
        
        # Initialize Gmail service
        self.gmail_service = get_gmail_service()
        if not self.gmail_service.authenticate():
            raise Exception("Gmail authentication failed")
        
# Initialize additional components
        self.memory_manager = memory_manager
        self.tool_registry = ToolRegistry()
        self.htn_planner = HTNPlanner(goal="Email Processing", task_descriptions={
            "Fetch Emails": "Fetch emails from Gmail",
            "Process Emails": "Process fetched emails"
        })
        self.mcts_planner = MCTSPlanner()
        self.task_executor = TaskExecutor()

        # Log current status
        last_processed = self._get_last_processed_date()
        self.logger.info(f"Last processed email date: {last_processed}")
        
        # Subscribe to fetch request events
        await event_bus.subscribe("email.fetch.request", self._handle_fetch_request)
        
    async def _handle_fetch_request(self, event):
        """Handle email fetch requests"""
        max_emails = event.data.get('max_emails', 10)
        self.logger.info(f"Processing fetch request for max {max_emails} emails")
        
        await self.process_emails_sequentially(max_emails)
    
    async def process_emails_sequentially(self, max_emails: int = 10):
        """
        Process emails in proper chronological order starting from Jan 1, 2025.
        This ensures proper chronological processing and timestamp management.
        """
        
        # Generate task plan
        task_plan = self.htn_planner.decompose_tasks()
        optimal_plan = self.mcts_planner.plan(initial_task=task_plan.tasks[0])
        execution_results = self.task_executor.execute_plan(task_plan)

        self.logger.info(f"Task execution completed: {execution_results}")

        # Store plan and results in memory
        await self.memory_manager.store_episode(
            agent_id=self.agent_id,
            task_type="Email Fetch Plan",
            context={"max_emails": max_emails},
            actions_taken=[{"task": task.description} for task in optimal_plan],
            outcome={"execution_results": execution_results},
            success=len(execution_results['failed_tasks']) == 0
        )
        try:
            # Get last processed date (not last update time)
            last_processed = self._get_last_processed_date()
            self.logger.info(f"Fetching emails since: {last_processed}")
            self.logger.info(f"Hard cutoff date: {self.CUTOFF_DATE}")
            
            # Search for Medium emails
            emails = self.gmail_service.mcp_service.search_medium_emails(last_processed)
            
            if not emails:
                self.logger.info("No new emails found")
                return {"success": True, "processed_emails": 0, "total_articles": 0}
            
            self.logger.info(f"Fetched {len(emails)} emails from Gmail")
            
            # CRITICAL: Filter and sort emails chronologically
            valid_emails = []
            for email in emails:
                email_date = email['date']
                
                # Skip emails before cutoff date (safety check)
                # Ensure both dates are timezone-aware for comparison
                cutoff_date = self.CUTOFF_DATE
                if email_date.tzinfo is None:
                    email_date = email_date.replace(tzinfo=timezone.utc)
                if cutoff_date.tzinfo is None:
                    cutoff_date = cutoff_date.replace(tzinfo=timezone.utc)
                
                if email_date < cutoff_date:
                    self.logger.warning(f"Skipping email from {email_date} (before cutoff)")
                    continue
                
                # Skip emails we've already processed
                # Ensure last_processed is timezone-aware
                last_proc = last_processed
                if last_proc.tzinfo is None:
                    last_proc = last_proc.replace(tzinfo=timezone.utc)
                
                if email_date <= last_proc:
                    self.logger.debug(f"Skipping email from {email_date} (already processed)")
                    continue
                
                valid_emails.append(email)
            
            # Limit emails if specified (take oldest first)
            if max_emails > 0 and len(valid_emails) > max_emails:
                valid_emails.sort(key=lambda x: x['date'])  # Sort first
                valid_emails = valid_emails[:max_emails]   # Take oldest
                self.logger.info(f"Limited to {max_emails} oldest emails for processing")
            
            # Sort emails chronologically (oldest first) for processing
            valid_emails.sort(key=lambda x: x['date'])
            
            if valid_emails:
                earliest = valid_emails[0]['date']
                latest = valid_emails[-1]['date']
                self.logger.info(f"Processing {len(valid_emails)} emails from {earliest} to {latest}")
            else:
                self.logger.info("No new emails to process after filtering")
                return {"success": True, "processed_emails": 0, "total_articles": 0}
            
            emails = valid_emails  # Use filtered emails
            
            total_articles = 0
            processed_count = 0
            
            # Process each email sequentially but articles in parallel
            for email in emails:
                try:
                    self.logger.info(f"Processing email from {email['date']}: {email['subject'][:50]}...")
                    
                    # Extract articles from this email
                    articles = self.gmail_service.mcp_service.get_articles_from_email(email)
                    
                    if articles:
                        # Process articles from this email in parallel
                        await self._process_articles_parallel(articles, email['date'])
                        total_articles += len(articles)
                        
                        self.logger.info(f"Processed {len(articles)} articles from email in parallel")
                    
                    # CRITICAL: Update timestamp after processing each email
                    # This ensures we don't reprocess emails if the system crashes
                    self._save_last_update_time(email['date'])
                    processed_count += 1
                    
                    self.logger.info(f"Updated last processed time to: {email['date']}")
                    
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing email {email.get('id', 'unknown')}: {e}")
                    continue
            
            # Publish processing complete event
            await event_bus.publish(
                event_type="email.processing.complete",
                data={
                    "processed_emails": processed_count,
                    "total_articles": total_articles,
                    "last_processed_date": emails[-1]['date'].isoformat() if emails else None
                },
                source=self.agent_id
            )
            
            self.logger.info(f"Email processing complete: {processed_count} emails, {total_articles} articles")
            
            return {
                "success": True,
                "processed_emails": processed_count,
                "total_articles": total_articles
            }
            
        except Exception as e:
            self.logger.error(f"Error in email processing: {e}")
            
            # Publish error event
            await event_bus.publish(
                event_type="email.processing.failed",
                data={
                    "error": str(e),
                    "failed_at": datetime.now().isoformat()
                },
                source=self.agent_id
            )
            
            return {"success": False, "error": str(e)}
    
    async def _process_articles_parallel(self, articles: List[Dict[str, Any]], email_date: datetime):
        """
        Process all articles from a single email in parallel with proper concurrency control.
        """
        if not articles:
            return
        
        # Create semaphore to limit concurrent processing (respect Medium rate limits)
        max_concurrent = 5  # Adjust based on your needs and Medium's tolerance
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_article(article):
            async with semaphore:
                try:
                    # Add email context to article (ensure JSON serializable)
                    article_data = dict(article)  # Make a copy
                    article_data['email_date'] = email_date.isoformat() if hasattr(email_date, 'isoformat') else str(email_date)
                    article_data['batch_id'] = f"email_{email_date.strftime('%Y%m%d_%H%M%S')}"
                    
                    # Ensure all date fields are JSON serializable
                    for key, value in article_data.items():
                        if hasattr(value, 'isoformat'):  # datetime object
                            article_data[key] = value.isoformat()
                        elif hasattr(value, 'strftime'):  # date object
                            article_data[key] = value.isoformat()
                    
                    # Publish article for processing
                    await event_bus.publish(
                        event_type="article.discovered",
                        data=article_data,
                        source=self.agent_id,
                        correlation_id=f"batch_{article_data['batch_id']}_{article_data['hash']}"
                    )
                    
                    self.logger.debug(f"Published article: {article['title'][:50]}...")
                    
                except Exception as e:
                    self.logger.error(f"Error processing article {article.get('title', 'Unknown')}: {e}")
        
        # Process all articles in parallel with controlled concurrency
        await asyncio.gather(
            *[process_single_article(article) for article in articles],
            return_exceptions=True
        )
        
        self.logger.info(f"Published {len(articles)} articles for parallel processing")
    
    def _get_last_update_time(self) -> datetime:
        """Get last update time from file"""
        timestamp_file = "/app/data/last_update.txt"
        try:
            if os.path.exists(timestamp_file):
                with open(timestamp_file, 'r') as f:
                    timestamp_str = f.read().strip()
                    return datetime.fromisoformat(timestamp_str)
            else:
                # Default to January 1, 2025 for first run (our cutoff date)
                return datetime(2025, 1, 1, 0, 0, 0)
        except Exception as e:
            self.logger.warning(f"Error reading timestamp file: {e}, using January 1, 2025")
            return datetime(2025, 1, 1, 0, 0, 0)
    
    def _get_last_processed_date(self) -> datetime:
        """Get the date of the last successfully processed email."""
        try:
            if os.path.exists(self.timestamp_file):
                with open(self.timestamp_file, 'r') as f:
                    timestamp_str = f.read().strip()
                    last_date = datetime.fromisoformat(timestamp_str)
                    
                    # Ensure both dates are timezone-aware for comparison
                    if last_date.tzinfo is None:
                        last_date = last_date.replace(tzinfo=timezone.utc)
                    
                    cutoff_date = self.CUTOFF_DATE
                    if cutoff_date.tzinfo is None:
                        cutoff_date = cutoff_date.replace(tzinfo=timezone.utc)
                    
                    # Ensure we never go before cutoff date
                    if last_date < cutoff_date:
                        self.logger.warning(f"Last processed date {last_date} is before cutoff, using cutoff")
                        return cutoff_date
                    
                    return last_date
            else:
                self.logger.info(f"No timestamp file found, starting from cutoff date: {self.CUTOFF_DATE}")
                return self.CUTOFF_DATE
                
        except Exception as e:
            self.logger.error(f"Error reading timestamp file: {e}, using cutoff date")
            return self.CUTOFF_DATE

    def _save_last_processed_date(self, email_date: datetime):
        """Save the date of the last successfully processed email."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.timestamp_file), exist_ok=True)
            
            # Only update if this email is newer than our last processed date
            current_last = self._get_last_processed_date()
            if email_date > current_last:
                with open(self.timestamp_file, 'w') as f:
                    f.write(email_date.isoformat())
                
                self.logger.info(f"Updated last processed date to: {email_date}")
            else:
                self.logger.debug(f"Email date {email_date} is not newer than last processed {current_last}")
                
        except Exception as e:
            self.logger.error(f"Error saving processed date: {e}")

    def _save_last_update_time(self, timestamp: datetime):
        """Save last update time to file (legacy compatibility)"""
        # Use the new chronological method
        self._save_last_processed_date(timestamp)
    
    async def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        last_update = self._get_last_update_time()
        
        return {
            "agent_id": self.agent_id,
            "status": "ready",
            "last_processed_time": last_update.isoformat(),
            "gmail_authenticated": self.gmail_service is not None
        }

# Usage example:
email_processor_agent = EmailProcessorAgent("email_processor_001")
