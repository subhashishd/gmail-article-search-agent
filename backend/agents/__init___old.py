"""
Multi-Agent Architecture for Gmail Article Search

This module implements a parallel, non-blocking multi-agent system with:
1. GmailFetcherAgent - Handles Gmail fetching and article extraction
2. SearchAgent - Handles complex search queries with RAG (works with available data)
3. ContentAnalysisAgent - Deep content analysis using LLMs
4. LLMCoordinatorAgent - Manages all LLM interactions
5. WorkflowOrchestrationAgent - Coordinates multi-agent workflows
6. MultiAgentCoordinator - Manages parallel, independent operations

Architecture Principles:
- Parallel, non-blocking operations
- Search works immediately with available data
- Gmail fetching runs independently in background
- Event-driven communication
- Agent autonomy and specialization
- LangChain integration for standardized patterns

Key Features:
- Search operates immediately without waiting for complete data fetching
- Background Gmail processing continuously updates the database
- Real-time article availability for search
- Independent agent lifecycles
"""

from .base_agent import BaseAgent, AgentMessage, AgentResponse
from .gmail_fetcher_agent import GmailFetcherAgent
from .search_agent import SearchAgent
from .content_analysis_agent import ContentAnalysisAgent
from .llm_coordinator_agent import LLMCoordinatorAgent
from .workflow_orchestration_agent import WorkflowOrchestrationAgent
from .multi_agent_coordinator import MultiAgentCoordinator, multi_agent_coordinator

__all__ = [
    "BaseAgent",
    "AgentMessage", 
    "AgentResponse",
    "GmailFetcherAgent",
    "SearchAgent",
    "ContentAnalysisAgent",
    "LLMCoordinatorAgent",
    "WorkflowOrchestrationAgent",
    "MultiAgentCoordinator",
    "multi_agent_coordinator"
]

__version__ = "1.0.0"
