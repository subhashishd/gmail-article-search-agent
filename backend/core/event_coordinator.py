"""
Event Coordinator for managing the new event-driven architecture.
Coordinates communication between all agents without being a bottleneck.
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from backend.core.event_bus import event_bus, Event
from backend.core.rate_limiter import rate_limiter
from backend.agents.email_processor_agent import EmailProcessorAgent
from backend.agents.content_agent import ContentAgent
from backend.agents.search_agent import SearchAgent

class EventCoordinator:
    """
    Event Coordinator for managing agent lifecycle and coordination.
    Acts as a lightweight orchestrator without being a bottleneck.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("EventCoordinator")
        
        # Agent instances - will be created during initialization
        self.email_processor = None
        self.content_agent = None
        self.search_agent = None
        
        # System state
        self.running = False
        self.scheduler_task = None
        self.last_fetch_time = None
        self.fetch_interval_hours = 24  # Fetch every 24 hours
        self.stats = {
            "emails_processed": 0,
            "articles_discovered": 0,
            "articles_stored": 0,
            "searches_performed": 0,
            "errors": 0
        }
    
    async def initialize(self):
        """Initialize all components"""
        self.logger.info("Initializing Event Coordinator...")
        
        try:
            # Initialize Redis-based event bus
            await event_bus.initialize()
            
            # Initialize rate limiter
            await rate_limiter.initialize()
            
            # Create agents now that Redis is available
            self.email_processor = EmailProcessorAgent("email_processor_001")
            self.content_agent = ContentAgent("content_agent_001", max_workers=3)
            self.search_agent = SearchAgent("search_agent_001")
            
            # Initialize agents
            await self.email_processor.initialize()
            await self.content_agent.initialize()
            await self.search_agent.initialize()
            
            # Subscribe to monitoring events
            await self._setup_monitoring()
            
            # Start background scheduler
            await self._start_background_scheduler()
            
            self.running = True
            self.logger.info("Event Coordinator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Event Coordinator: {e}")
            raise
    
    async def _setup_monitoring(self):
        """Setup event monitoring for statistics"""
        await event_bus.subscribe("email.processing.complete", self._track_emails_processed)
        await event_bus.subscribe("article.discovered", self._track_articles_discovered)
        await event_bus.subscribe("article.stored", self._track_articles_stored)
        await event_bus.subscribe("search.completed", self._track_searches_performed)
        await event_bus.subscribe("*.failed", self._track_errors)  # Any failure event
    
    async def _track_emails_processed(self, event: Event):
        """Track email processing stats"""
        processed_count = event.data.get("processed_emails", 0)
        self.stats["emails_processed"] += processed_count
        self.logger.info(f"Tracked {processed_count} emails processed")
    
    async def _track_articles_discovered(self, event: Event):
        """Track article discovery stats"""
        self.stats["articles_discovered"] += 1
    
    async def _track_articles_stored(self, event: Event):
        """Track article storage stats"""
        self.stats["articles_stored"] += 1
    
    async def _track_searches_performed(self, event: Event):
        """Track search stats"""
        self.stats["searches_performed"] += 1
    
    async def _track_errors(self, event: Event):
        """Track error stats"""
        self.stats["errors"] += 1
        self.logger.warning(f"Error tracked: {event.type} - {event.data.get('error', 'Unknown error')}")
    
    async def _start_background_scheduler(self):
        """Start background scheduler for automatic email fetching"""
        self.logger.info(f"Starting background scheduler - fetch every {self.fetch_interval_hours} hours")
        self.scheduler_task = asyncio.create_task(self._background_fetch_loop())
    
    async def _background_fetch_loop(self):
        """Background loop that fetches emails every 24 hours"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check if it's time to fetch (if never fetched or 24+ hours passed)
                should_fetch = (
                    self.last_fetch_time is None or 
                    current_time - self.last_fetch_time >= timedelta(hours=self.fetch_interval_hours)
                )
                
                if should_fetch:
                    self.logger.info("Background scheduler triggering email fetch")
                    
                    # Trigger fetch with a reasonable batch size
                    result = await self.trigger_email_fetch(max_emails=20)
                    
                    if result.get("success"):
                        self.last_fetch_time = current_time
                        self.logger.info(f"Background fetch completed successfully at {current_time}")
                    else:
                        self.logger.error(f"Background fetch failed: {result.get('error', 'Unknown error')}")
                
                # Sleep for 1 hour before checking again
                await asyncio.sleep(3600)  # 1 hour
                
            except asyncio.CancelledError:
                self.logger.info("Background scheduler cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in background scheduler: {e}")
                # Sleep for 30 minutes on error before retrying
                await asyncio.sleep(1800)
    
    # Public API methods
    
    async def trigger_email_fetch(self, max_emails: int = 10) -> Dict[str, Any]:
        """Trigger email fetching process"""
        try:
            self.logger.info(f"Triggering email fetch for max {max_emails} emails")
            
            # Publish fetch request event
            event_id = await event_bus.publish(
                event_type="email.fetch.request",
                data={"max_emails": max_emails, "triggered_at": datetime.now().isoformat()},
                source="event_coordinator"
            )
            
            return {
                "success": True,
                "message": f"Email fetch triggered for max {max_emails} emails",
                "event_id": event_id,
                "status": "processing"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to trigger email fetch: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_articles(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Perform article search with caching"""
        try:
            self.logger.info(f"Performing search: {query}")
            
            # Use search agent directly for immediate response
            result = await self.search_agent.search_and_cache(query, top_k)
            
            # Track search
            await event_bus.publish(
                event_type="search.completed",
                data={"query": query, "results_count": len(result.get("results", []))},
                source="event_coordinator"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {"results": [], "total_found": 0, "error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get actual database stats for accurate reporting
            actual_stats = await self._get_actual_system_stats()
            
            # Get agent statuses
            email_status = await self.email_processor.get_processing_status()
            content_status = await self.content_agent.get_queue_status()
            
            # Get rate limiter status
            rate_status = {
                "medium": await rate_limiter.get_status("medium"),
                "default": await rate_limiter.get_status("default")
            }
            
            # Get recent events
            recent_events = await event_bus.get_event_history(limit=10)
            
            # Determine current operation based on system activity
            current_operation = self._determine_current_operation(content_status, email_status)
            
            return {
                "coordinator_status": "running" if self.running else "stopped",
                "current_operation": current_operation,
                "stats": self.stats.copy(),
                "agents": {
                    "email_processor": email_status,
                    "content_agent": content_status,
                    "search_agent": {"status": "ready"}
                },
                "rate_limiting": rate_status,
                "background_scheduler": {
                    "running": self.scheduler_task is not None and not self.scheduler_task.done(),
                    "last_fetch_time": self.last_fetch_time.isoformat() if self.last_fetch_time else None,
                    "next_fetch_in_hours": self._get_hours_until_next_fetch(),
                    "fetch_interval_hours": self.fetch_interval_hours
                },
                "background_tasks": {
                    "email_fetching": {
                        "active": self.scheduler_task is not None and not self.scheduler_task.done(),
                        "last_run": self.last_fetch_time.isoformat() if self.last_fetch_time else None
                    }
                },
                "recent_events": [
                    {
                        "type": event.type,
                        "timestamp": event.timestamp.isoformat(),
                        "source": event.source
                    } for event in recent_events
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def _get_actual_system_stats(self) -> Dict[str, Any]:
        """Get actual database statistics to supplement coordinator stats"""
        try:
            from backend.services.hybrid_rag_service import hybrid_rag_service
            db_stats = hybrid_rag_service.get_article_stats()
            
            # If database has more articles than coordinator, update coordinator stats
            db_articles = db_stats.get("total_articles", 0)
            if db_articles > self.stats["articles_stored"]:
                self.logger.info(f"Updating coordinator stats: database has {db_articles} articles, coordinator had {self.stats['articles_stored']}")
                self.stats["articles_stored"] = db_articles
                # Also update discovered count to be at least as high as stored
                if self.stats["articles_discovered"] < db_articles:
                    self.stats["articles_discovered"] = db_articles
            
            return {
                "database_articles": db_articles,
                "coordinator_articles": self.stats["articles_stored"],
                "synced": True
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get database stats: {e}")
            return {
                "database_articles": 0,
                "coordinator_articles": self.stats["articles_stored"],
                "synced": False
            }
    
    def _determine_current_operation(self, content_status: Dict[str, Any], email_status: Dict[str, Any]) -> str:
        """Determine current operation based on agent activity with detailed status"""
        try:
            # Check content processing status
            content_queue = content_status.get("queue_size", 0)
            active_workers = content_status.get("active_workers", 0)
            total_workers = content_status.get("total_workers", 0)
            
            # Check email processing status
            email_processing = email_status.get("status", "idle") == "processing"
            
            # Check background scheduler status
            scheduler_running = self.scheduler_task is not None and not self.scheduler_task.done()
            
            # Get article processing stats
            articles_stored = self.stats.get("articles_stored", 0)
            articles_discovered = self.stats.get("articles_discovered", 0)
            
            # Priority order: email processing > active content processing > queued content > scheduler > idle
            if email_processing:
                return "Fetching and Processing Emails"
            elif content_queue > 0 and active_workers > 0:
                return f"Processing Articles ({active_workers}/{total_workers} workers active, {content_queue} in queue)"
            elif content_queue > 0:
                return f"Articles Queued for Processing ({content_queue} pending)"
            elif articles_stored > 0 and articles_discovered > articles_stored:
                remaining = articles_discovered - articles_stored
                return f"Article Processing ({articles_stored} processed, ~{remaining} remaining)"
            elif scheduler_running:
                next_fetch_hours = self._get_hours_until_next_fetch()
                return f"Monitoring for New Emails (next check in {next_fetch_hours:.1f}h)"
            elif articles_stored > 0:
                return f"System Ready ({articles_stored} articles available for search)"
            else:
                return "System Ready"
                
        except Exception as e:
            self.logger.error(f"Error determining current operation: {e}")
            return "Unknown"
    
    def _get_hours_until_next_fetch(self) -> float:
        """Calculate hours until next scheduled fetch"""
        if self.last_fetch_time is None:
            return 0.0  # Fetch immediately
        
        current_time = datetime.now()
        next_fetch_time = self.last_fetch_time + timedelta(hours=self.fetch_interval_hours)
        time_diff = next_fetch_time - current_time
        
        return max(0.0, time_diff.total_seconds() / 3600)  # Convert to hours
    
    async def clear_search_cache(self) -> Dict[str, Any]:
        """Clear search cache"""
        try:
            await self.search_agent.clear_cache()
            return {"success": True, "message": "Search cache cleared"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        self.logger.info("Shutting down Event Coordinator...")
        
        self.running = False
        
        # Cancel background scheduler
        if self.scheduler_task and not self.scheduler_task.done():
            self.logger.info("Cancelling background scheduler")
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        try:
            # Stop agents
            await self.content_agent.stop()
            
            # Cleanup infrastructure
            await rate_limiter.cleanup()
            await event_bus.cleanup()
            
            self.logger.info("Event Coordinator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Global coordinator instance
event_coordinator = EventCoordinator()
