"""
Gmail Agent for fetching articles from Gmail.
Fetches Medium Daily Digest emails in the background and publishes events.
"""

import asyncio
import logging
from backend.core.event_bus import event_bus
from backend.services.gmail_service_oauth import GmailService

class GmailAgent:
    """Agent for handling Gmail fetching tasks."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"GmailAgent-{self.agent_id}")
        self.gmail_service = GmailService()
        self.fetch_interval = 300  # 5 minutes
    
    async def initialize(self):
        """Initialize Gmail Agent resources."""
        self.logger.info("Initializing Gmail Agent...")
        await event_bus.subscribe("gmail.fetch.request", self._handle_fetch_request)
        
    async def run(self):
        """Continuously fetch emails every fetch_interval."""
        while True:
            try:
                await self.fetch_and_publish_articles()
                await asyncio.sleep(self.fetch_interval)
            except Exception as e:
                self.logger.error(f"Error during fetch and publish: {e}")
                await asyncio.sleep(self.fetch_interval)
    
    async def fetch_and_publish_articles(self):
        """Fetch emails and publish articles as events."""
        self.logger.info("Fetching Medium Daily Digest emails...")
        articles = await self.gmail_service.fetch_medium_digests()
        
        for article in articles:
            await event_bus.publish(
                event_type="article.discovered",
                data=article,
                source=self.agent_id
            )
            self.logger.info(f"Published article event: {article['title']}")
    
    async def _handle_fetch_request(self, event):
        """Handle fetch requests from event bus."""
        self.logger.info(f"Received fetch request event: {event.id}")
        await self.fetch_and_publish_articles()

# Usage example:
gmail_agent = GmailAgent("gmail_agent_001")
