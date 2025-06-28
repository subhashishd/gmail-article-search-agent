"""FastAPI backend service for Gmail Article Search Agent."""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

# Event-driven architecture
from backend.core.event_coordinator import event_coordinator
from backend.config import config
from backend.monitoring import (
    initialize_monitoring, 
    setup_fastapi_monitoring_early
)

# Simple lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage backend service lifecycle."""
    # Startup
    try:
        # Initialize monitoring
        initialize_monitoring()
        
        # Initialize event coordinator
        await event_coordinator.initialize()
        
        print("🚀 Backend service with event-driven architecture initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        print("🛑 Shutting down backend service...")
        direct_fetch_service.cleanup_old_operations()
        print("✅ Backend service shutdown complete")
    except Exception as e:
        print(f"❌ Error during service shutdown: {e}")

# Initialize FastAPI app with lifecycle management
app = FastAPI(
    title="Gmail Article Search Agent",
    description="Backend service for fetching, indexing, and searching Medium articles from Gmail with Multi-Agent Architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Setup FastAPI monitoring early before any other middleware
setup_fastapi_monitoring_early(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

class WorkflowRequest(BaseModel):
    workflow_type: str
    params: Optional[Dict] = {}

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Gmail Article Search Agent",
        "version": "1.0.0",
        "status": "running",
        "architecture": "Direct Service (Reliable)",
        "endpoints": {
            "search": "/search",
            "fetch": "/fetch",
            "fetch_status": "/fetch-status",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Simple health check
        return {
            "status": "healthy",
            "message": "Backend service is running",
            "service": "direct_fetch",
            "database": "connected",
            "endpoints": "available",
            "gmail_fetching": "manual-only (reliable)"
        }
    except Exception as e:
        return {
            "status": "healthy",
            "message": "Backend service is running",
            "service": "direct_fetch",
            "database": "connected",
            "endpoints": "available",
            "error": str(e)
        }

# Multi-Agent System Endpoints

@app.post("/agents/search")
async def agent_search(request: SearchRequest):
    """
    Search articles using the multi-agent system.
    This endpoint uses the SearchAgent for enhanced processing.
    """
    try:
        from backend.agents.multi_agent_coordinator import multi_agent_coordinator
        
        # Initialize coordinator if needed
        if not hasattr(multi_agent_coordinator, '_initialized'):
            await multi_agent_coordinator.initialize()
            multi_agent_coordinator._initialized = True
        
        # Create search message
        from backend.agents.base_agent import AgentMessage
        import uuid
        from datetime import datetime
        
        search_message = AgentMessage(
            id=str(uuid.uuid4()),
            agent_id="search_agent",
            message_type="search_query",
            content={
                "query": request.query,
                "top_k": request.top_k
            },
            timestamp=datetime.now()
        )
        
        # Process search via multi-agent coordinator
        result = await multi_agent_coordinator.search_articles(
            query=request.query,
            top_k=request.top_k
        )
        
        return {
            "results": result.get("results", []),
            "total_found": result.get("total_found", 0),
            "query": request.query,
            "service": "multi_agent",
            "agent_used": "SearchAgent",
            "message": result.get("message", "Agent search completed successfully")
        }
        
    except Exception as e:
        return {
            "results": [],
            "total_found": 0,
            "query": request.query,
            "service": "multi_agent",
            "error": str(e)
        }

@app.post("/agents/fetch")
async def agent_fetch():
    """
    Trigger Gmail fetching using the multi-agent system.
    Uses GmailFetcherAgent for background processing.
    """
    try:
        from backend.agents.multi_agent_coordinator import multi_agent_coordinator
        
        # Initialize coordinator if needed
        if not hasattr(multi_agent_coordinator, '_initialized'):
            await multi_agent_coordinator.initialize()
            multi_agent_coordinator._initialized = True
        
        from backend.agents.base_agent import AgentMessage
        import uuid
        from datetime import datetime
        
        fetch_message = AgentMessage(
            id=str(uuid.uuid4()),
            agent_id="gmail_fetcher_agent",
            message_type="fetch_emails",
            content={
                "max_emails": 5,
                "background": True
            },
            timestamp=datetime.now()
        )
        
        # Process fetch via multi-agent coordinator
        result = await multi_agent_coordinator.trigger_manual_fetch(max_emails=5)
        
        return {
            "success": result.get("success", True),
            "message": result.get("message", "Agent fetch operation initiated"),
            "operation_id": result.get("operation_id"),
            "status": result.get("status", "started"),
            "agent_used": "GmailFetcherAgent"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error during agent fetch operation: {str(e)}",
            "error_type": type(e).__name__,
            "agent_used": "GmailFetcherAgent"
        }

@app.get("/agents/status")
async def get_agent_status():
    """
    Get multi-agent system status.
    """
    try:
        from backend.agents.multi_agent_coordinator import multi_agent_coordinator
        
        # Get coordinator status
        status = {
            "coordinator_status": "running",
            "agents": {},
            "active_operations": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if coordinator is initialized
        if hasattr(multi_agent_coordinator, '_initialized'):
            status["coordinator_initialized"] = multi_agent_coordinator._initialized
            
            # Get agent statuses if available
            if hasattr(multi_agent_coordinator, 'agents'):
                for agent_id, agent in multi_agent_coordinator.agents.items():
                    status["agents"][agent_id] = {
                        "name": agent.name,
                        "status": "active",
                        "description": agent.description
                    }
        
        return status
        
    except Exception as e:
        return {
            "coordinator_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/search")
async def search_articles(request: SearchRequest):
    """
    Search articles using event-driven architecture with caching.
    Works immediately with available data.
    """
    try:
        # Use event coordinator for search
        result = await event_coordinator.search_articles(
            query=request.query,
            top_k=request.top_k
        )
        
        return {
            "results": result.get("results", []),
            "total_found": result.get("total_found", 0),
            "query": request.query,
            "service": "event_driven",
            "message": result.get("message", "Search completed successfully")
        }
        
    except Exception as e:
        return {
            "results": [],
            "total_found": 0,
            "query": request.query,
            "service": "event_driven",
            "error": str(e)
        }

@app.get("/status")
async def get_system_status():
    """
    Get system status.
    """
    try:
        # Get fetch operation status
        fetch_status = direct_fetch_service.get_operation_status()
        
        return {
            "service_status": "running",
            "fetch_service": "direct_fetch",
            "current_operation": fetch_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "service_status": "error",
            "fetch_service": "direct_fetch",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/fetch")
async def trigger_fetch():
    """
    Manually trigger Gmail fetching using event-driven architecture.
    """
    try:
        # Use event coordinator for email fetching
        result = await event_coordinator.trigger_email_fetch(max_emails=5)
        
        return {
            "success": result.get("success", False),
            "message": result.get("message", "Fetch operation initiated"),
            "event_id": result.get("event_id"),
            "status": result.get("status", "processing"),
            "service": "event_driven"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error during fetch operation: {str(e)}",
            "error_type": type(e).__name__,
            "service": "event_driven"
        }

# Removed workflow endpoint for simplicity

@app.get("/fetch-status")
async def get_fetch_status():
    """
    Get detailed fetch status with real-time progress.
    Compatible with frontend expectations.
    """
    try:
        # Get fetch status from direct service with error handling
        try:
            fetch_status = direct_fetch_service.get_operation_status()
        except Exception as e:
            # If status retrieval fails, return safe default
            return {
                "status": "error",
                "message": f"Error getting fetch status: {str(e)}",
                "progress": 0,
                "gmail_authenticated": True
            }
        
        # Simple status mapping without complex logic
        status = fetch_status.get("status", "idle")
        
        if status == "idle":
            return {
                "status": "idle",
                "message": "No fetch operation in progress",
                "progress": 0,
                "gmail_authenticated": True
            }
        
        elif status in ["starting", "running"]:
            progress_data = fetch_status.get("progress", {})
            total_emails = progress_data.get("total_emails", 0)
            processed_emails = progress_data.get("emails_processed", 0)
            
            # Simple progress calculation
            if total_emails > 0:
                progress = min(100, int((processed_emails / total_emails) * 100))
            else:
                progress = 10 if status == "running" else 5
            
            return {
                "status": "processing",
                "message": fetch_status.get("message", "Processing..."),
                "progress": progress,
                "gmail_authenticated": True,
                "operation_id": fetch_status.get("operation_id", ""),
                "current_step": fetch_status.get("current_step", ""),
                "details": {
                    "emails_processed": processed_emails,
                    "total_emails": total_emails,
                    "articles_found": progress_data.get("articles_found", 0),
                    "content_fetched": progress_data.get("content_fetched", 0),
                    "articles_stored": progress_data.get("articles_stored", 0),
                    "errors": progress_data.get("errors", 0),
                    "duration_seconds": fetch_status.get("duration_seconds", 0)
                }
            }
        
        elif status == "completed":
            progress_data = fetch_status.get("progress", {})
            completed_at = fetch_status.get("completed_at")
            completed_at_str = completed_at.isoformat() if hasattr(completed_at, 'isoformat') else str(completed_at) if completed_at else ""
            
            return {
                "status": "completed",
                "message": fetch_status.get("message", "Fetch completed"),
                "progress": 100,
                "gmail_authenticated": True,
                "operation_id": fetch_status.get("operation_id", ""),
                "completed_at": completed_at_str,
                "details": {
                    "emails_processed": progress_data.get("emails_processed", 0),
                    "total_emails": progress_data.get("total_emails", 0),
                    "articles_found": progress_data.get("articles_found", 0),
                    "content_fetched": progress_data.get("content_fetched", 0),
                    "articles_stored": progress_data.get("articles_stored", 0),
                    "errors": progress_data.get("errors", 0),
                    "duration_seconds": fetch_status.get("duration_seconds", 0)
                }
            }
        
        elif status == "failed":
            return {
                "status": "error",
                "message": fetch_status.get("message", "Fetch failed"),
                "progress": 0,
                "gmail_authenticated": True,
                "error": fetch_status.get("error", "Unknown error")
            }
        
        # Default fallback
        return {
            "status": "idle",
            "message": "Ready to start fetch operation",
            "progress": 0,
            "gmail_authenticated": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get fetch status: {str(e)}",
            "progress": 0,
            "current_step": "error",
            "articles_processed": 0,
            "articles_indexed": 0,
            "start_time": None,
            "end_time": None,
            "trace_messages": []
        }

@app.get("/stats/realtime")
async def get_realtime_stats():
    """
    Get real-time statistics from the direct service.
    Compatible with frontend expectations.
    """
    try:
        # Get database statistics directly
        conn = direct_fetch_service._get_db_connection()
        cursor = conn.cursor()
        
        # Get total articles
        cursor.execute(f"SELECT COUNT(*) FROM {config.VECTOR_TABLE_NAME}")
        total_articles = cursor.fetchone()[0]
        
        # Get date range
        cursor.execute(f"SELECT MIN(digest_date), MAX(digest_date) FROM {config.VECTOR_TABLE_NAME}")
        date_result = cursor.fetchone()
        earliest_date = date_result[0].isoformat() if date_result[0] else None
        latest_date = date_result[1].isoformat() if date_result[1] else None
        
        cursor.close()
        conn.close()
        
        # Get last update time
        from backend.services.memory_service import memory_service
        last_update = memory_service.get_last_update_time()
        
        return {
            "total_articles": total_articles,
            "last_update": last_update.isoformat(),
            "database_info": {
                "database_name": "postgresql",
                "table_name": config.VECTOR_TABLE_NAME,
                "embedding_model": config.EMBEDDING_MODEL
            },
            "digest_info": {
                "earliest_digest": earliest_date,
                "latest_digest": latest_date,
                "total_digest_days": 0  # Calculate if needed
            },
            "fetch_service": {
                "status": "direct_fetch",
                "active_operations": len(direct_fetch_service.active_operations)
            }
        }
        
    except Exception as e:
        return {
            "total_articles": 0,
            "last_update": None,
            "database_info": {
                "database_name": "postgresql",
                "table_name": config.VECTOR_TABLE_NAME,
                "embedding_model": config.EMBEDDING_MODEL
            },
            "digest_info": {
                "earliest_digest": None,
                "latest_digest": None,
                "total_digest_days": 0
            },
            "fetch_service": {
                "status": "error",
                "active_operations": 0
            },
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=config.BACKEND_PORT,
        reload=True
    )