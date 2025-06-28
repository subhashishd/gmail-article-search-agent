"""
Search Agent for handling complex query processing.
Implements Hybrid RAG (Retrieval-Augmented Generation) with vector search + LLM contextualization

Responsibilities:
- Process user search queries using hybrid vector + LLM approach
- Retrieve relevant articles from the database using vector similarity
- Use LLM to analyze and re-rank results for better relevance
- Provide enhanced search results with AI insights

Architecture:
- Vector search for initial candidate retrieval
- LLM analysis for relevance scoring and context synthesis
- Combined scoring and ranking for optimal results
"""

import asyncio
import json
import logging
import os
from datetime import datetime, date
import redis.asyncio as redis
from typing import List, Dict, Any
from .base_agent import BaseAgent, AgentMessage, AgentResponse
from backend.services.hybrid_rag_service import hybrid_rag_service
from backend.services.ollama_service import ollama_service
from backend.monitoring import monitor_agent_operation
from backend.services.rag_evaluation_service import rag_evaluation_service
from backend.core.rate_limiter import RateLimitedRequest, rate_limiter

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

class SearchAgent(BaseAgent):
    """
    Agent for handling enhanced article search requests.
    """
    
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            name="SearchAgent",
            description="Agent for enhanced RAG-based search and retrieval"
        )
        self.logger = logging.getLogger(f"SearchAgent-{self.agent_id}")
        self.cache_ttl = 3600  # Cache TTL in seconds
        self.redis_client = None
        
    async def initialize(self):
        """Initialize search agent and Redis connection."""
        self.logger.info("Initializing Search Agent...")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
    
    async def cleanup(self):
        """Cleanup search agent resources."""
        if self.redis_client:
            await self.redis_client.close()
        self.logger.info("Search Agent cleanup complete")
    
    async def handle_query(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle query messages - delegate to search functionality."""
        query = message.content.get("query", "")
        top_k = message.content.get("top_k", 10)
        
        if not query:
            return {"error": "Query is required"}
        
        # Perform search
        search_results = await self.search_and_cache(query, top_k)
        return search_results
        
    async def search_and_cache(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Search articles and cache results for frequent queries."""
        self.logger.info(f"Searching for query: {query}")
        cache_key = f"search_cache:{query}:{top_k}"
        
        # Check cache first
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            self.logger.info(f"Cache hit for query: {query}")
            return json.loads(cached_result)
        
        # Perform search with rate limiting
        async with RateLimitedRequest(rate_limiter, "default", timeout=10.0):
            result = await hybrid_rag_service.search_and_analyze(query, top_k)
        
        # Cache result
        await self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(result, cls=DateTimeEncoder))
        self.logger.info(f"Search completed with {len(result.get('results', []))} results for query: {query}")
        return result
        
    async def clear_cache(self):
        """Clear the search cache."""
        await self.redis_client.flushdb()
        self.logger.info("Cache cleared")
        
    async def search_articles(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a search to retrieve articles matching the query using RAG pipeline.
        """
        self.logger.info(f"Initiating search in database for query: {query}")
        # TODO: Implement vector search and retrieval logic
        return []  # Placeholder for actual search logic
        
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics using the hybrid RAG service.
        """
        try:
            return hybrid_rag_service.get_article_stats()
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {
                "total_articles": 0,
                "error": str(e)
            }
            
    def generate_response(self, query: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response based on retrieved articles and LLM analysis.
        """
        self.logger.info(f"Generating response for query: {query} with {len(articles)} articles")
        # TODO: Implement LLM response generation
        return {
            "query": query,
            "results": [],  # Placeholder
            "total_found": 0  # Placeholder
        }
        
    async def handle_search(self, message: AgentMessage) -> AgentResponse:
        """
        Handle search requests from the event coordinator.
        """
        with monitor_agent_operation(self.name, "search"):
            try:
                query = message.data.get("query", "")
                top_k = message.data.get("top_k", 10)
                
                if not query:
                    return AgentResponse(
                        status="error",
                        data={"error": "Query is required"}
                    )
                
                # Perform hybrid RAG search
                search_results = await self.search_and_cache(query, top_k)
                
                return AgentResponse(
                    status="success",
                    data=search_results
                )
                
            except Exception as e:
                self.logger.error(f"Error handling search query: {e}")
                return AgentResponse(
                    status="error",
                    data={
                        "results": [],
                        "total_found": 0,
                        "query": message.data.get("query", ""),
                        "error": str(e)
                    }
                )
