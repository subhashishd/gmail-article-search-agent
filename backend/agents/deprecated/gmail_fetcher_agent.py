"""
Gmail Fetcher Agent

This agent handles all Gmail-related operations:
- Fetches emails from Gmail using OAuth2 authentication
- Extracts Medium article links and metadata from emails
- Processes emails chronologically with proper timestamp tracking
- Handles email content parsing and article extraction
- Manages authentication and API rate limiting

Inter-agent Communication:
- Provides article data to ContentAnalysisAgent for processing
- Coordinates with WorkflowOrchestrationAgent for article processing workflows
- Updates memory service with last processed timestamps
"""

import logging
import hashlib
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_agent import BaseAgent, AgentMessage, AgentResponse, MessageType
from backend.monitoring import (
    monitor_agent_operation,
    record_article_processing,
    update_system_metrics
)

class GmailFetcherAgent(BaseAgent):
    """
    Agent for fetching and processing emails from Gmail.
    """

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            name="GmailFetcherAgent",
            description="Agent for fetching Medium articles from Gmail emails"
        )
        self.logger = logging.getLogger(f"GmailFetcherAgent-{self.agent_id}")
        self.gmail_service = None
        self.memory_service = None
        self.content_fetcher = None

    async def initialize(self):
        """Initialize Gmail service and dependencies."""
        self.logger.info("Initializing Gmail Fetcher Agent...")
        
        try:
            # Initialize Gmail OAuth2 service (lazy loading to prevent automatic processing)
            self.gmail_service = None
            
            # Initialize memory service for timestamp tracking
            from backend.services.memory_service import memory_service
            self.memory_service = memory_service
            
            # Initialize content fetcher (lazy loading)
            self.content_fetcher = None
            
            # Note: Authentication will be performed lazily when needed
            # This prevents blocking during system startup
            self.logger.info("Gmail service initialized (authentication will be performed on-demand)")
                
        except Exception as e:
            self.logger.error(f"Error initializing Gmail Fetcher Agent: {e}")

    async def cleanup(self):
        """Cleanup Gmail resources."""
        self.logger.info("Cleaning up Gmail Fetcher Agent resources...")

    async def handle_query(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle Gmail fetching requests."""
        request_type = message.content.get('request_type')
        
        if request_type == 'fetch_new_emails':
            return await self._fetch_new_emails(message)
        elif request_type == 'fetch_articles_from_email':
            return await self._fetch_articles_from_email(message)
        elif request_type == 'get_last_update_time':
            return await self._get_last_update_time(message)
        elif request_type == 'update_last_processed_time':
            return await self._update_last_processed_time(message)
        elif request_type == 'fetch_article_content':
            return await self._fetch_article_content(message)
        elif request_type == 'process_emails_chronologically':
            return await self._process_emails_chronologically(message)
        elif request_type == 'bulk_fetch_emails_with_articles':
            return await self._bulk_fetch_emails_with_articles(message)
        else:
            return {"error": f"Unknown request type: {request_type}"}

    async def _fetch_new_emails(self, message: AgentMessage) -> Dict[str, Any]:
        """Fetch new emails from Gmail since last update."""
        try:
            # Lazy load Gmail service when needed
            if self.gmail_service is None:
                from backend.services.gmail_service_oauth import get_gmail_service
                self.gmail_service = get_gmail_service()
            
            last_update = self.memory_service.get_last_update_time()
            self.logger.info(f"Fetching emails since: {last_update}")
            
            # Search for Medium emails
            emails = self.gmail_service.mcp_service.search_medium_emails(last_update=last_update)
            
            if not emails:
                return {
                    "success": True,
                    "message": "No new emails found",
                    "emails": [],
                    "count": 0
                }
            
            # Sort emails chronologically (oldest first)
            sorted_emails = sorted(emails, key=lambda m: m['date'], reverse=False)
            
            self.logger.info(f"Found {len(sorted_emails)} new emails")
            
            return {
                "success": True,
                "message": f"Found {len(sorted_emails)} new emails",
                "emails": sorted_emails,
                "count": len(sorted_emails),
                "date_range": {
                    "from": sorted_emails[0]['date'].isoformat() if sorted_emails else None,
                    "to": sorted_emails[-1]['date'].isoformat() if sorted_emails else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching emails: {e}")
            return {
                "success": False,
                "error": str(e),
                "emails": [],
                "count": 0
            }

    async def _fetch_articles_from_email(self, message: AgentMessage) -> Dict[str, Any]:
        """Extract articles from a specific email."""
        try:
            email_content = message.content.get('email_content')
            if not email_content:
                return {
                    "success": False,
                    "error": "No email content provided",
                    "articles": []
                }
            
            # Extract articles from email
            articles = self.gmail_service.mcp_service.get_articles_from_email(email_content)
            
            self.logger.info(f"Extracted {len(articles)} articles from email")
            
            return {
                "success": True,
                "message": f"Extracted {len(articles)} articles",
                "articles": articles,
                "count": len(articles)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting articles from email: {e}")
            return {
                "success": False,
                "error": str(e),
                "articles": []
            }

    async def _get_last_update_time(self, message: AgentMessage) -> Dict[str, Any]:
        """Get the last update time from memory service."""
        try:
            last_update = self.memory_service.get_last_update_time()
            
            return {
                "success": True,
                "last_update_time": last_update.isoformat(),
                "timestamp": last_update
            }
            
        except Exception as e:
            self.logger.error(f"Error getting last update time: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _update_last_processed_time(self, message: AgentMessage) -> Dict[str, Any]:
        """Update the last processed time in memory service."""
        try:
            new_time = message.content.get('timestamp')
            if isinstance(new_time, str):
                new_time = datetime.fromisoformat(new_time)
            elif not isinstance(new_time, datetime):
                return {
                    "success": False,
                    "error": "Invalid timestamp format"
                }
            
            self.memory_service.save_last_update_time(new_time)
            
            self.logger.info(f"Updated last processed time to: {new_time}")
            
            return {
                "success": True,
                "message": f"Updated last processed time to {new_time.isoformat()}",
                "timestamp": new_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating last processed time: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _fetch_article_content(self, message: AgentMessage) -> Dict[str, Any]:
        """Fetch full content from a Medium article URL."""
        try:
            article_url = message.content.get('url')
            if not article_url:
                return {
                    "success": False,
                    "error": "No article URL provided"
                }
            
            # Lazy load content fetcher
            if self.content_fetcher is None:
                from backend.services.article_processing_service import ArticleContentFetcher
                self.content_fetcher = ArticleContentFetcher()
            
            # Fetch full content
            content = self.content_fetcher.fetch_article_content(article_url)
            
            return {
                "success": True,
                "url": article_url,
                "content": content,
                "content_length": len(content)
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching article content: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": article_url
            }

    async def _process_emails_chronologically(self, message: AgentMessage) -> Dict[str, Any]:
        """Process emails chronologically with full workflow integration."""
        with monitor_agent_operation(self.name, "process_emails_chronologically"):
            try:
                # Get processing parameters
                max_emails = message.content.get('max_emails', None)
                include_content_fetching = message.content.get('include_content_fetching', True)
                
                # Fetch new emails
                fetch_result = await self._fetch_new_emails(message)
                if not fetch_result.get('success'):
                    return fetch_result
                
                emails = fetch_result.get('emails', [])
                
                if not emails:
                    return {
                        "success": True,
                        "message": "No emails to process",
                        "processed_count": 0,
                        "total_articles": 0
                    }
                
                # Limit emails if specified
                if max_emails:
                    emails = emails[:max_emails]
                
                self.logger.info(f"Processing {len(emails)} emails chronologically")
                
                processed_emails = 0
                total_articles = 0
                processed_articles = []
                
                # Process each email chronologically
                for email_idx, email in enumerate(emails, 1):
                    email_date = email['date']
                    self.logger.info(f"Processing email {email_idx}/{len(emails)} from {email_date}")
                    
                    # Extract articles from email
                    articles_result = await self._fetch_articles_from_email(
                        AgentMessage(
                            id=str(uuid.uuid4()),
                            agent_id=self.agent_id,
                            message_type=MessageType.QUERY,
                            content={'email_content': email},
                            timestamp=datetime.now()
                        )
                    )
                    
                    if articles_result.get('success'):
                        articles = articles_result.get('articles', [])
                        total_articles += len(articles)
                        
                        # Process each article
                        for article in articles:
                            article['digest_date'] = email_date.date()
                            
                            # Fetch full content if requested
                            if include_content_fetching:
                                content_result = await self._fetch_article_content(
                                    AgentMessage(
                                        id=str(uuid.uuid4()),
                                        agent_id=self.agent_id,
                                        message_type=MessageType.QUERY,
                                        content={'url': article['link']},
                                        timestamp=datetime.now()
                                    )
                                )
                                
                                if content_result.get('success'):
                                    article['full_content'] = content_result.get('content', '')
                                    record_article_processing("fetch_content", "success")
                                else:
                                    article['full_content'] = f"Unable to fetch content: {content_result.get('error', 'Unknown error')}"
                                    record_article_processing("fetch_content", "error")
                            
                            # Record article extraction
                            record_article_processing("extract", "success")
                            processed_articles.append(article)
                    
                    # Update last processed time after each email
                    await self._update_last_processed_time(
                        AgentMessage(
                            id=str(uuid.uuid4()),
                            agent_id=self.agent_id,
                            message_type=MessageType.QUERY,
                            content={'timestamp': email_date},
                            timestamp=datetime.now()
                        )
                    )
                    
                    processed_emails += 1
                
                result_message = f"Successfully processed {processed_emails} emails with {total_articles} articles"
                self.logger.info(result_message)
                
                return {
                    "success": True,
                    "message": result_message,
                    "processed_emails": processed_emails,
                    "total_articles": total_articles,
                    "articles": processed_articles,
                    "include_content_fetching": include_content_fetching
                }
                
            except Exception as e:
                self.logger.error(f"Error in chronological email processing: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "processed_emails": 0,
                    "total_articles": 0
                }

    async def _bulk_fetch_emails_with_articles(self, message: AgentMessage) -> Dict[str, Any]:
        """
        NEW ARCHITECTURE: Bulk fetch all emails and extract article metadata in-memory.
        
        This method:
        1. Fetches all Gmail emails at once (fast Gmail API operation)
        2. Extracts article metadata from each email (fast, in-memory parsing)
        3. Returns a complete article dataset for parallel processing
        
        No database operations or content fetching - pure in-memory data preparation.
        """
        with monitor_agent_operation(self.name, "bulk_fetch_emails_with_articles"):
            try:
                max_emails = message.content.get('max_emails', None)
                
                # Step 1: Bulk fetch all emails from Gmail (single API call)
                self.logger.info("Starting bulk Gmail fetch...")
                fetch_result = await self._fetch_new_emails(message)
                
                if not fetch_result.get('success'):
                    return fetch_result
                
                emails = fetch_result.get('emails', [])
                if not emails:
                    return {
                        "success": True,
                        "message": "No new emails found",
                        "articles_dataset": [],
                        "total_emails": 0,
                        "latest_email_date": None
                    }
                
                # Limit emails if specified
                if max_emails:
                    emails = emails[:max_emails]
                
                self.logger.info(f"Bulk processing {len(emails)} emails for article extraction...")
                
                # Step 2: Fast in-memory article extraction from all emails
                articles_dataset = []
                total_articles = 0
                latest_email_date = None
                
                for email_idx, email in enumerate(emails, 1):
                    try:
                        email_date = email['date']
                        
                        # Track latest email date
                        if latest_email_date is None or email_date > latest_email_date:
                            latest_email_date = email_date
                        
                        # Extract articles from email (fast, in-memory operation)
                        articles_result = await self._fetch_articles_from_email(
                            AgentMessage(
                                id=str(uuid.uuid4()),
                                agent_id=self.agent_id,
                                message_type=MessageType.QUERY,
                                content={'email_content': email},
                                timestamp=datetime.now()
                            )
                        )
                        
                        if articles_result.get('success'):
                            articles = articles_result.get('articles', [])
                            total_articles += len(articles)
                            
                            # Add digest date to each article
                            for article in articles:
                                article['digest_date'] = email_date.date()
                                article['email_index'] = email_idx
                                article['email_date'] = email_date.isoformat()
                                # Note: full_content will be fetched later in parallel
                                article['full_content'] = None  # Placeholder for parallel fetching
                            
                            articles_dataset.extend(articles)
                        
                        # Log progress every 10 emails
                        if email_idx % 10 == 0:
                            self.logger.info(f"Processed {email_idx}/{len(emails)} emails, found {total_articles} articles")
                    
                    except Exception as e:
                        self.logger.error(f"Error processing email {email_idx}: {e}")
                        continue
                
                self.logger.info(f"Bulk extraction complete: {len(emails)} emails processed, {len(articles_dataset)} articles extracted")
                
                return {
                    "success": True,
                    "message": f"Bulk extracted {len(articles_dataset)} articles from {len(emails)} emails",
                    "articles_dataset": articles_dataset,
                    "total_emails": len(emails),
                    "total_articles": len(articles_dataset),
                    "latest_email_date": latest_email_date,
                    "processing_time": "bulk_extraction_only"  # Content fetching happens later in parallel
                }
                
            except Exception as e:
                self.logger.error(f"Error in bulk fetch: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "articles_dataset": [],
                    "total_emails": 0
                }

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get Gmail agent statistics."""
        try:
            last_update = self.memory_service.get_last_update_time()
            auth_status = "authenticated" if self.gmail_service else "not_authenticated"
            
            return {
                "agent_name": self.name,
                "agent_id": self.agent_id,
                "authentication_status": auth_status,
                "last_update_time": last_update.isoformat(),
                "capabilities": [
                    "fetch_new_emails",
                    "fetch_articles_from_email", 
                    "fetch_article_content",
                    "process_emails_chronologically",
                    "bulk_fetch_emails_with_articles",  # New capability
                    "timestamp_management"
                ]
            }
            
        except Exception as e:
            return {
                "agent_name": self.name,
                "agent_id": self.agent_id,
                "error": str(e)
            }
