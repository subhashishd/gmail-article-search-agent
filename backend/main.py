"""FastAPI backend service for Gmail Article Search Agent."""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# Event-driven architecture
from backend.core.event_coordinator import event_coordinator
from backend.config import config
from backend.monitoring import (
    initialize_monitoring, 
    setup_fastapi_monitoring_early
)

# Lifecycle management
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
        await event_coordinator.shutdown()
        print("✅ Backend service shutdown complete")
    except Exception as e:
        print(f"❌ Error during service shutdown: {e}")

# Initialize FastAPI app with lifecycle management
app = FastAPI(
    title="Gmail Article Search Agent",
    description="Event-driven backend service for fetching, indexing, and searching Medium articles from Gmail",
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

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Gmail Article Search Agent",
        "version": "1.0.0",
        "status": "running",
        "architecture": "Event-Driven Multi-Agent",
        "endpoints": {
            "search": "/search",
            "fetch": "/fetch",
            "health": "/health",
            "status": "/status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with agent information."""
    try:
        # Get system status to include agent counts
        system_status = await event_coordinator.get_system_status()
        
        agents = system_status.get("agents", {})
        agent_count = len([a for a in agents.values() if isinstance(a, dict) and a.get("status") == "ready" or a.get("running")])
        
        # Count background tasks
        scheduler_running = system_status.get("background_scheduler", {}).get("running", False)
        content_queue = agents.get("content_agent", {}).get("queue_size", 0)
        background_tasks = (1 if scheduler_running else 0) + (1 if content_queue > 0 else 0)
        
        return {
            "status": "healthy",
            "message": "Backend service is running",
            "service": "event_driven",
            "database": "connected",
            "endpoints": "available",
            "gmail_fetching": "event-driven",
            "agents_count": agent_count,
            "background_tasks": background_tasks,
            "processing_queue_size": content_queue
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Backend service error: {str(e)}",
            "service": "event_driven",
            "error": str(e),
            "agents_count": 0,
            "background_tasks": 0
        }

@app.post("/search")
async def search_articles(request: SearchRequest):
    """Search articles using event-driven architecture with caching."""
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
            "message": "Search completed successfully"
        }
        
    except Exception as e:
        return {
            "results": [],
            "total_found": 0,
            "query": request.query,
            "service": "event_driven",
            "error": str(e)
        }

@app.post("/fetch")
async def trigger_fetch():
    """Manually trigger Gmail fetching using event-driven architecture."""
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

@app.get("/status")
async def get_system_status():
    """Get comprehensive system status."""
    try:
        # Get system status from event coordinator
        status = await event_coordinator.get_system_status()
        
        return {
            "service_status": "running",
            "architecture": "event_driven",
            "coordinator": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "service_status": "error",
            "architecture": "event_driven",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/stats/realtime")
async def get_realtime_stats():
    """Get real-time statistics."""
    try:
        # Get system status which includes statistics
        status = await event_coordinator.get_system_status()
        
        # Get actual database statistics from the hybrid RAG service
        from backend.services.hybrid_rag_service import hybrid_rag_service
        db_stats = hybrid_rag_service.get_article_stats()
        
        return {
            "total_articles": db_stats.get("total_articles", 0),
            "last_update": datetime.now().isoformat(),
            "database_info": {
                "database_name": config.DB_NAME,
                "table_name": config.VECTOR_TABLE_NAME,
                "embedding_model": "all-MiniLM-L6-v2"
            },
            "service": {
                "architecture": "event_driven",
                "status": "running"
            },
            "coordinator_stats": status.get("stats", {}),
            "earliest_digest": db_stats.get("earliest_digest"),
            "latest_digest": db_stats.get("latest_digest"),
            "total_digest_days": db_stats.get("total_digest_days", 0),
            "timestamp": datetime.now().isoformat()
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
            "service": {
                "architecture": "event_driven",
                "status": "error"
            },
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/fetch-status")
async def get_fetch_status():
    """Get fetch operation status."""
    try:
        # Get system status from event coordinator
        status = await event_coordinator.get_system_status()
        
        # Get content agent queue status
        content_agent = status.get("agents", {}).get("content_agent", {})
        queue_size = content_agent.get("queue_size", 0)
        active_workers = content_agent.get("active_workers", 0)
        
        # Get actual database stats for accurate reporting
        from backend.services.hybrid_rag_service import hybrid_rag_service
        try:
            db_stats = hybrid_rag_service.get_article_stats()
            actual_articles_stored = db_stats.get("total_articles", 0)
        except:
            actual_articles_stored = 0
        
        # Get coordinator stats (may be reset after restart)
        stats = status.get("stats", {})
        coordinator_articles = stats.get("articles_stored", 0)
        
        # Use the higher of database count or coordinator count for accuracy
        articles_processed = max(actual_articles_stored, coordinator_articles)
        
        # Get background scheduler info
        bg_scheduler = status.get("background_scheduler", {})
        scheduler_running = bg_scheduler.get("running", False)
        last_fetch_time = bg_scheduler.get("last_fetch_time")
        
        # Determine current status based on activity
        if queue_size > 0 and active_workers > 0:
            current_status = "processing"
            message = f"Processing articles - Queue: {queue_size}, Active workers: {active_workers}"
            current_step = "article_processing"
            # Calculate progress based on processed vs queue
            total_work = articles_processed + queue_size
            progress = int((articles_processed / max(total_work, 1)) * 100) if total_work > 0 else 0
        elif queue_size > 0:
            current_status = "processing"
            message = f"Articles queued for processing - Queue: {queue_size}"
            current_step = "article_processing"
            progress = int((articles_processed / max(articles_processed + queue_size, 1)) * 100)
        elif articles_processed > 0 and scheduler_running:
            current_status = "completed"
            message = f"Processing complete - {articles_processed} articles indexed. Monitoring for new emails."
            current_step = "email_monitoring"
            progress = 100
        elif scheduler_running:
            current_status = "monitoring"
            message = "Background scheduler active - monitoring for new emails"
            current_step = "email_monitoring"
            progress = 100
        elif articles_processed > 0:
            current_status = "completed"
            message = f"Processing complete - {articles_processed} articles available for search"
            current_step = "completed"
            progress = 100
        else:
            current_status = "idle" 
            message = "System ready - no articles processed yet"
            current_step = "idle"
            progress = 0
        
        # Calculate timing information
        current_time = datetime.now()
        if last_fetch_time:
            try:
                last_fetch_dt = datetime.fromisoformat(last_fetch_time.replace('Z', '+00:00'))
                runtime_seconds = (current_time - last_fetch_dt.replace(tzinfo=None)).total_seconds()
                runtime = f"{int(runtime_seconds // 60)}m {int(runtime_seconds % 60)}s"
            except:
                runtime = "Unknown"
        else:
            runtime = "--"
        
        # Estimate completion based on processing rate
        if queue_size > 0 and active_workers > 0:
            # Estimate ~30 seconds per article with 3 workers
            estimated_seconds = (queue_size / max(active_workers, 1)) * 30
            remaining_time = f"{int(estimated_seconds // 60)}m {int(estimated_seconds % 60)}s"
            eta = (current_time + timedelta(seconds=estimated_seconds)).strftime("%H:%M")
        else:
            remaining_time = "--"
            eta = "--"
        
        # Get recent events for trace messages
        recent_events = status.get("recent_events", [])
        trace_messages = []
        
        # Convert recent events to trace messages
        for event in recent_events[-10:]:  # Last 10 events
            event_type = event.get("type", "")
            timestamp = event.get("timestamp", "")
            source = event.get("source", "")
            
            if "article" in event_type:
                if event_type == "article.stored":
                    msg = f"✅ Article successfully stored"
                elif event_type == "article.discovered":
                    msg = f"🔍 New article discovered"
                elif event_type == "article.processing.failed":
                    msg = f"❌ Article processing failed"
                else:
                    msg = f"📊 Article event: {event_type}"
                    
                trace_messages.append({
                    "timestamp": timestamp,
                    "message": msg
                })
        
        return {
            "status": current_status,
            "message": message,
            "progress": progress,
            "current_step": current_step,
            "articles_processed": articles_processed,
            "articles_indexed": articles_processed,
            "queue_size": queue_size,
            "active_workers": active_workers,
            "start_time": last_fetch_time,
            "end_time": None,
            "runtime": runtime,
            "remaining_time": remaining_time,
            "eta": eta,
            "completed": articles_processed,
            "trace_messages": trace_messages,
            "service": "event_driven"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting fetch status: {str(e)}",
            "progress": 0,
            "service": "event_driven"
        }

@app.post("/cache/clear")
async def clear_cache():
    """Clear search cache."""
    try:
        result = await event_coordinator.clear_search_cache()
        return {
            "success": result.get("success", False),
            "message": result.get("message", "Cache cleared"),
            "service": "event_driven"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error clearing cache: {str(e)}",
            "service": "event_driven"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=config.BACKEND_PORT,
        reload=True
    )
