"""
Multi-Agent System Coordinator

This coordinator manages a parallel, non-blocking multi-agent architecture where:
- Search operations work immediately with available data
- Gmail fetching runs independently in background
- Content analysis enhances articles asynchronously
- No blocking dependencies between major operations

Architecture Principles:
- Asynchronous, parallel execution
- Event-driven communication
- Independent agent lifecycles
- Real-time data availability
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_agent import BaseAgent, AgentMessage, AgentResponse, MessageType
from .gmail_fetcher_agent import GmailFetcherAgent
from .search_agent import SearchAgent
from .content_analysis_agent import ContentAnalysisAgent
from .llm_coordinator_agent import LLMCoordinatorAgent
from .workflow_orchestration_agent import WorkflowOrchestrationAgent

class MultiAgentCoordinator:
    """
    Coordinator for managing parallel, independent multi-agent operations.
    """

    def __init__(self):
        self.logger = logging.getLogger("MultiAgentCoordinator")
        self.agents = {}
        self.background_tasks = {}
        self.event_queue = asyncio.Queue()
        self.running = False

    async def initialize(self):
        """Initialize all agents with minimal overhead to prevent hanging."""
        self.logger.info("Initializing Multi-Agent System...")
        
        try:
            # Initialize core agents with lazy loading
            await self._initialize_agents_lazy()
            
            # Start with minimal background operations
            self.running = True
            self.logger.info("Multi-Agent System initialized successfully (lazy mode)")
            
        except Exception as e:
            self.logger.error(f"Error initializing Multi-Agent System: {e}")
            raise

    async def _initialize_agents_lazy(self):
        """Initialize agents with lazy loading to prevent hanging."""
        
        # Initialize only critical agents first
        self.logger.info("Initializing core agents...")
        
        # Gmail Fetcher Agent - handles email fetching
        gmail_agent = GmailFetcherAgent("gmail_fetcher_001")
        await gmail_agent.initialize()
        self.agents['gmail_fetcher'] = gmail_agent
        self.logger.info("Gmail Fetcher Agent initialized")
        
        # Search Agent - handles search operations
        search_agent = SearchAgent("search_agent_001")
        await search_agent.initialize()
        self.agents['search'] = search_agent
        self.logger.info("Search Agent initialized")
        
        # Content Analysis Agent - defer heavy initialization
        self.agents['content_analysis'] = None  # Lazy load when needed
        
        # LLM Coordinator Agent - defer heavy initialization
        self.agents['llm_coordinator'] = None  # Lazy load when needed
        
        # Workflow Orchestration Agent - minimal initialization
        workflow_agent = WorkflowOrchestrationAgent("workflow_orchestrator_001")
        await workflow_agent.initialize()
        self.agents['workflow_orchestrator'] = workflow_agent
        self.logger.info("Workflow Orchestration Agent initialized")
        
        self.logger.info("Core agents initialized successfully")

    def _ensure_agent_loaded(self, agent_name: str):
        """Ensure an agent is loaded, initializing if needed."""
        if self.agents.get(agent_name) is None:
            if agent_name == 'content_analysis':
                from .content_analysis_agent import ContentAnalysisAgent
                agent = ContentAnalysisAgent(f"{agent_name}_001")
                # Note: we'll initialize this synchronously when needed
                self.agents[agent_name] = agent
            elif agent_name == 'llm_coordinator':
                from .llm_coordinator_agent import LLMCoordinatorAgent
                agent = LLMCoordinatorAgent(f"{agent_name}_001")
                # Note: we'll initialize this synchronously when needed
                self.agents[agent_name] = agent
        
        return self.agents[agent_name]

    async def _start_background_operations(self):
        """Start independent background operations."""
        
        # NOTE: Gmail fetching is now manual-only, triggered via API calls
        # No automatic background fetching to prevent unwanted processing
        
        # Background content enhancement (processes articles asynchronously)
        self.background_tasks['content_enhancement'] = asyncio.create_task(
            self._background_content_enhancement()
        )
        
        # Event processing loop
        self.background_tasks['event_processing'] = asyncio.create_task(
            self._process_events()
        )
        
        self.logger.info("Started background operations (Gmail fetching is manual-only)")

    async def _background_gmail_fetching(self):
        """Continuously fetch emails in background without blocking search."""
        gmail_agent = self.agents['gmail_fetcher']
        
        while self.running:
            try:
                self.logger.info("Background: Checking for new emails...")
                
                # Fetch new emails (non-blocking)
                result = await gmail_agent.handle_query(AgentMessage(
                    id=str(uuid.uuid4()),
                    agent_id="coordinator",
                    message_type=MessageType.QUERY,
                    content={
                        'request_type': 'process_emails_chronologically',
                        'max_emails': 10,  # Process in batches
                        'include_content_fetching': True
                    },
                    timestamp=datetime.now()
                ))
                
                if result.get('success'):
                    processed_emails = result.get('processed_emails', 0)
                    total_articles = result.get('total_articles', 0)
                    
                    if processed_emails > 0:
                        self.logger.info(f"Background: Processed {processed_emails} emails, {total_articles} articles")
                        
                        # Emit event for processed articles
                        await self.event_queue.put({
                            'type': 'articles_processed',
                            'data': {
                                'processed_emails': processed_emails,
                                'total_articles': total_articles,
                                'articles': result.get('articles', [])
                            }
                        })
                    else:
                        self.logger.info("Background: No new emails found")
                        # Wait longer if no new emails
                        await asyncio.sleep(300)  # 5 minutes
                        continue
                else:
                    self.logger.error(f"Background Gmail fetch error: {result.get('error')}")
                
                # Wait before next fetch cycle
                await asyncio.sleep(60)  # 1 minute between batches
                
            except asyncio.CancelledError:
                self.logger.info("Background Gmail fetching cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in background Gmail fetching: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    async def _background_content_enhancement(self):
        """Enhance articles with LLM analysis in background."""
        
        while self.running:
            try:
                # This could enhance articles that haven't been fully analyzed
                # For now, just wait and process events
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                self.logger.info("Background content enhancement cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in background content enhancement: {e}")
                await asyncio.sleep(30)

    async def _process_events(self):
        """Process system events from various agents."""
        
        while self.running:
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=5.0)
                    await self._handle_event(event)
                except asyncio.TimeoutError:
                    continue  # No events to process
                    
            except asyncio.CancelledError:
                self.logger.info("Event processing cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error processing events: {e}")

    async def _handle_event(self, event: Dict[str, Any]):
        """Handle system events."""
        event_type = event.get('type')
        event_data = event.get('data', {})
        
        if event_type == 'articles_processed':
            articles_count = event_data.get('total_articles', 0)
            self.logger.info(f"Event: {articles_count} new articles available for search")
            
        # Add more event handlers as needed
        self.logger.debug(f"Processed event: {event_type}")

    # Public API methods for external use

    async def search_articles(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Search articles immediately with whatever is available in database.
        Non-blocking operation that works independently of Gmail fetching.
        """
        try:
            search_agent = self.agents['search']
            
            # Direct search - no waiting for fetching to complete
            result = await search_agent.handle_query(AgentMessage(
                id=str(uuid.uuid4()),
                agent_id="coordinator",
                message_type=MessageType.QUERY,
                content={
                    'request_type': 'search_and_analyze',
                    'query': query,
                    'top_k': top_k
                },
                timestamp=datetime.now()
            ))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in search: {e}")
            return {"error": str(e), "results": [], "total_found": 0}

    async def get_system_status(self) -> Dict[str, Any]:
        """Get current status of all agents and background operations."""
        try:
            status = {
                "coordinator_status": "running" if self.running else "stopped",
                "agents": {},
                "background_tasks": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Get agent statuses
            for agent_name, agent in self.agents.items():
                try:
                    if hasattr(agent, 'get_agent_stats'):
                        status["agents"][agent_name] = agent.get_agent_stats()
                    else:
                        status["agents"][agent_name] = {
                            "agent_id": agent.agent_id,
                            "name": agent.name,
                            "status": "active"
                        }
                except Exception as e:
                    status["agents"][agent_name] = {"error": str(e)}
            
            # Get background task statuses
            for task_name, task in self.background_tasks.items():
                status["background_tasks"][task_name] = {
                    "running": not task.done(),
                    "cancelled": task.cancelled(),
                    "exception": str(task.exception()) if task.done() and task.exception() else None
                }
            
            return status
            
        except Exception as e:
            return {"error": str(e)}

    async def start_workflow(self, workflow_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start a specific workflow through the orchestrator."""
        try:
            workflow_agent = self.agents['workflow_orchestrator']
            
            result = await workflow_agent.handle_query(AgentMessage(
                id=str(uuid.uuid4()),
                agent_id="coordinator",
                message_type=MessageType.QUERY,
                content={
                    'request_type': 'start_workflow',
                    'workflow_type': workflow_type,
                    'params': params
                },
                timestamp=datetime.now()
            ))
            
            return result
            
        except Exception as e:
            return {"error": str(e)}

    async def trigger_manual_fetch(self, max_emails: int = 5) -> Dict[str, Any]:
        """Manually trigger email fetching - returns immediately, runs in background."""
        try:
            # Check if fetch is already running
            if hasattr(self, 'active_fetch_task') and not self.active_fetch_task.done():
                return {
                    "success": False,
                    "message": "Fetch operation already in progress",
                    "status": "running",
                    "fetch_id": getattr(self, 'current_fetch_id', 'unknown')
                }
            
            # Generate unique fetch ID
            fetch_id = f"fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.current_fetch_id = fetch_id
            
            # Initialize fetch state
            self.fetch_state = {
                "fetch_id": fetch_id,
                "status": "starting",
                "started_at": datetime.now(),
                "progress": {
                    "emails_processed": 0,
                    "articles_found": 0,
                    "articles_processed": 0,
                    "content_fetched": 0,
                    "errors": 0
                },
                "current_step": "initializing",
                "message": "Starting fetch operation..."
            }
            
            # Start background fetch task
            self.active_fetch_task = asyncio.create_task(
                self._background_fetch_process(max_emails)
            )
            
            # Give it a moment to start
            await asyncio.sleep(0.1)
            
            return {
                "success": True,
                "message": "Fetch operation started in background",
                "status": "running",
                "fetch_id": fetch_id,
                "estimated_duration": "1-3 minutes"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _background_fetch_process(self, max_emails: int):
        """
        Background process for fetching and processing emails with progress tracking.
        
        New Architecture:
        1. Bulk fetch all Gmail emails and extract article metadata quickly (in-memory)
        2. Process the in-memory dataset with full parallelism for content fetching and DB storage
        """
        try:
            self.fetch_state["status"] = "running"
            self.fetch_state["current_step"] = "bulk_fetching_emails"
            self.fetch_state["message"] = "Bulk fetching emails and extracting article metadata..."
            
            gmail_agent = self.agents['gmail_fetcher']
            
            # Step 1: Bulk fetch all emails with article extraction (fast, in-memory operation)
            bulk_result = await gmail_agent.handle_query(AgentMessage(
                id=str(uuid.uuid4()),
                agent_id="coordinator",
                message_type=MessageType.QUERY,
                content={
                    'request_type': 'bulk_fetch_emails_with_articles',
                    'max_emails': max_emails
                },
                timestamp=datetime.now()
            ))
            
            if not bulk_result.get('success'):
                self.fetch_state["status"] = "failed"
                self.fetch_state["message"] = f"Failed to bulk fetch emails: {bulk_result.get('error')}"
                return
            
            # Get the in-memory article dataset
            articles_dataset = bulk_result.get('articles_dataset', [])
            total_emails = bulk_result.get('total_emails', 0)
            latest_email_date = bulk_result.get('latest_email_date')
            
            if not articles_dataset:
                self.fetch_state["status"] = "completed"
                self.fetch_state["message"] = "No new articles found"
                self.fetch_state["completed_at"] = datetime.now()
                return
            
            # Update progress with bulk results
            self.fetch_state["progress"]["total_emails"] = total_emails
            self.fetch_state["progress"]["emails_processed"] = total_emails
            self.fetch_state["progress"]["articles_found"] = len(articles_dataset)
            self.fetch_state["current_step"] = "parallel_processing"
            self.fetch_state["message"] = f"Processing {len(articles_dataset)} articles in parallel from {total_emails} emails..."
            
            self.logger.info(f"Bulk extracted {len(articles_dataset)} articles from {total_emails} emails")
            
            # Step 2: Parallel content fetching and database storage from in-memory dataset
            await self._parallel_content_processing(articles_dataset)
            
            # Step 3: Update timestamp after all processing is complete
            if latest_email_date:
                await gmail_agent.handle_query(AgentMessage(
                    id=str(uuid.uuid4()),
                    agent_id="coordinator",
                    message_type=MessageType.QUERY,
                    content={'request_type': 'update_last_processed_time', 'timestamp': latest_email_date},
                    timestamp=datetime.now()
                ))
            
            # Mark as completed
            self.fetch_state["status"] = "completed"
            self.fetch_state["completed_at"] = datetime.now()
            self.fetch_state["message"] = f"Successfully processed {total_emails} emails with {len(articles_dataset)} articles"
            
        except Exception as e:
            self.logger.error(f"Error in background fetch process: {e}")
            self.fetch_state["status"] = "failed"
            self.fetch_state["message"] = f"Fetch failed: {str(e)}"
            self.fetch_state["error"] = str(e)
    
    async def _parallel_content_processing(self, articles: List[Dict]):
        """Process article content fetching and database storage in parallel."""
        from backend.services.article_processing_service import store_articles_batch
        
        # Create semaphore to limit concurrent requests (prevent rate limiting)
        max_concurrent = 5  # Adjust based on Medium's rate limits
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_article(article):
            async with semaphore:
                try:
                    gmail_agent = self.agents['gmail_fetcher']
                    
                    # Fetch content
                    content_result = await gmail_agent.handle_query(AgentMessage(
                        id=str(uuid.uuid4()),
                        agent_id="coordinator",
                        message_type=MessageType.QUERY,
                        content={'request_type': 'fetch_article_content', 'url': article['link']},
                        timestamp=datetime.now()
                    ))
                    
                    if content_result.get('success'):
                        article['full_content'] = content_result.get('content', '')
                        self.fetch_state["progress"]["content_fetched"] += 1
                    else:
                        article['full_content'] = f"Unable to fetch: {content_result.get('error', 'Unknown error')}"
                        self.fetch_state["progress"]["errors"] += 1
                    
                    return article
                    
                except Exception as e:
                    self.logger.error(f"Error processing article {article.get('link', 'unknown')}: {e}")
                    article['full_content'] = f"Processing error: {str(e)}"
                    self.fetch_state["progress"]["errors"] += 1
                    return article
        
        # Process articles in parallel
        processed_articles = await asyncio.gather(
            *[process_single_article(article) for article in articles],
            return_exceptions=True
        )
        
        # Filter out exceptions and store in database
        valid_articles = [a for a in processed_articles if not isinstance(a, Exception)]
        
        if valid_articles:
            try:
                # Store articles in database in batch
                stored_count = await store_articles_batch(valid_articles)
                self.fetch_state["progress"]["articles_processed"] = stored_count
                self.logger.info(f"Stored {stored_count} articles in database")
            except Exception as e:
                self.logger.error(f"Error storing articles in database: {e}")
                self.fetch_state["progress"]["errors"] += 1
    
    def get_fetch_status(self) -> Dict[str, Any]:
        """Get current fetch operation status."""
        if not hasattr(self, 'fetch_state'):
            return {
                "status": "idle",
                "message": "No fetch operation in progress",
                "last_fetch": None
            }
        
        # Add duration calculation
        fetch_state = self.fetch_state.copy()
        if "started_at" in fetch_state:
            duration = (datetime.now() - fetch_state["started_at"]).total_seconds()
            fetch_state["duration_seconds"] = duration
        
        return fetch_state

    async def shutdown(self):
        """Gracefully shutdown the multi-agent system."""
        self.logger.info("Shutting down Multi-Agent System...")
        
        self.running = False
        
        # Cancel background tasks
        for task_name, task in self.background_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                self.logger.info(f"Cancelled {task_name}")
        
        # Cleanup agents
        for agent_name, agent in self.agents.items():
            try:
                await agent.cleanup()
                self.logger.info(f"Cleaned up {agent_name}")
            except Exception as e:
                self.logger.error(f"Error cleaning up {agent_name}: {e}")
        
        self.logger.info("Multi-Agent System shutdown complete")

# Global coordinator instance
multi_agent_coordinator = MultiAgentCoordinator()
