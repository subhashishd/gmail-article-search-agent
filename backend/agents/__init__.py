"""
Event-Driven Multi-Agent Architecture for Gmail Article Search

This module implements the core agents for the event-driven system:
1. EmailProcessorAgent - Handles Gmail fetching and email processing
2. ContentAgent - Processes and stores article content with parallel workers
3. SearchAgent - Handles complex search queries with RAG and caching

Architecture Principles:
- Event-driven communication via Redis
- Parallel, non-blocking operations
- Rate limiting and proper error handling
- Agent autonomy and specialization
- Proper separation of concerns

Key Features:
- Asynchronous email processing
- Parallel article content fetching with rate limiting
- Vector search with LLM enhancement
- Redis-based caching and event bus
"""

from .base_agent import BaseAgent, AgentMessage, AgentResponse
from .email_processor_agent import EmailProcessorAgent
from .content_agent import ContentAgent
from .search_agent import SearchAgent

__all__ = [
    "BaseAgent",
    "AgentMessage", 
    "AgentResponse",
    "EmailProcessorAgent",
    "ContentAgent", 
    "SearchAgent"
]

__version__ = "2.0.0"  # Event-driven architecture version
