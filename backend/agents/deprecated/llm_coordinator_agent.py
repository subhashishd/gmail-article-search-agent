"""
LLM Coordinator Agent

This agent manages all LLM interactions across the system:
- Coordinates requests to different LLM models (Ollama, local models)
- Handles prompt engineering and response parsing
- Manages LLM context and memory
- Provides unified LLM interface for other agents

Inter-agent Communication:
- Serves LLM requests from ContentAnalysisAgent, SearchAgent, and others
- Manages model selection based on task requirements
- Handles fallback strategies for model failures
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentMessage, AgentResponse

class LLMCoordinatorAgent(BaseAgent):
    """
    Agent for coordinating all LLM interactions in the system.
    """

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            name="LLMCoordinatorAgent",
            description="Central coordinator for all LLM interactions and model management"
        )
        self.logger = logging.getLogger(f"LLMCoordinatorAgent-{self.agent_id}")
        self.available_models = {}
        self.model_capabilities = {}
        self.default_model = "llama3.2:1b"

    async def initialize(self):
        """Initialize LLM models and check availability."""
        self.logger.info("Initializing LLM Coordinator Agent...")
        
        try:
            # Initialize Ollama service
            from backend.services.ollama_service import ollama_service
            self.ollama_service = ollama_service
            
            # Check available models
            await self._discover_available_models()
            
            self.logger.info(f"LLM Coordinator initialized with models: {list(self.available_models.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM Coordinator: {e}")

    async def cleanup(self):
        """Cleanup LLM resources."""
        self.logger.info("Cleaning up LLM Coordinator resources...")

    async def handle_query(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle LLM request messages from other agents."""
        request_type = message.content.get('request_type')
        
        if request_type == 'analyze_content':
            return await self._handle_content_analysis(message)
        elif request_type == 'search_query_enhancement':
            return await self._handle_search_enhancement(message)
        elif request_type == 'summarization':
            return await self._handle_summarization(message)
        elif request_type == 'classification':
            return await self._handle_classification(message)
        else:
            return await self._handle_generic_llm_request(message)

    async def _discover_available_models(self):
        """Discover and catalog available LLM models."""
        try:
            # Check Ollama models
            if hasattr(self.ollama_service, 'list_models'):
                models = await self.ollama_service.list_models()
                for model in models:
                    self.available_models[model] = 'ollama'
                    self.model_capabilities[model] = {
                        'type': 'conversational',
                        'context_window': 4096,  # Default, could be model-specific
                        'provider': 'ollama'
                    }
            
            if not self.available_models:
                # Fallback to default model
                self.available_models[self.default_model] = 'ollama'
                self.model_capabilities[self.default_model] = {
                    'type': 'conversational',
                    'context_window': 4096,
                    'provider': 'ollama'
                }
                
        except Exception as e:
            self.logger.error(f"Error discovering models: {e}")

    async def _handle_content_analysis(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle content analysis requests."""
        content = message.content.get('content', '')
        analysis_type = message.content.get('analysis_type', 'general')
        
        prompt = self._build_content_analysis_prompt(content, analysis_type)
        
        try:
            response = await self.ollama_service.generate_response(
                prompt=prompt,
                model=self.default_model
            )
            
            return {
                'analysis_result': response,
                'model_used': self.default_model,
                'analysis_type': analysis_type,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error in content analysis: {e}")
            return {
                'analysis_result': '',
                'error': str(e),
                'success': False
            }

    async def _handle_search_enhancement(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle search query enhancement requests."""
        query = message.content.get('query', '')
        context = message.content.get('context', '')
        
        prompt = self._build_search_enhancement_prompt(query, context)
        
        try:
            response = await self.ollama_service.generate_response(
                prompt=prompt,
                model=self.default_model
            )
            
            return {
                'enhanced_query': response,
                'original_query': query,
                'model_used': self.default_model,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error in search enhancement: {e}")
            return {
                'enhanced_query': query,  # Fallback to original
                'error': str(e),
                'success': False
            }

    async def _handle_summarization(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle summarization requests."""
        content = message.content.get('content', '')
        max_length = message.content.get('max_length', 200)
        
        prompt = self._build_summarization_prompt(content, max_length)
        
        try:
            response = await self.ollama_service.generate_response(
                prompt=prompt,
                model=self.default_model
            )
            
            return {
                'summary': response,
                'original_length': len(content),
                'summary_length': len(response),
                'model_used': self.default_model,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error in summarization: {e}")
            return {
                'summary': content[:max_length] + "...",  # Fallback
                'error': str(e),
                'success': False
            }

    async def _handle_classification(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle classification requests."""
        content = message.content.get('content', '')
        categories = message.content.get('categories', [])
        
        prompt = self._build_classification_prompt(content, categories)
        
        try:
            response = await self.ollama_service.generate_response(
                prompt=prompt,
                model=self.default_model
            )
            
            return {
                'classification': response,
                'categories': categories,
                'model_used': self.default_model,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error in classification: {e}")
            return {
                'classification': 'uncategorized',
                'error': str(e),
                'success': False
            }

    async def _handle_generic_llm_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle generic LLM requests."""
        prompt = message.content.get('prompt', '')
        model = message.content.get('model', self.default_model)
        
        try:
            response = await self.ollama_service.generate_response(
                prompt=prompt,
                model=model
            )
            
            return {
                'response': response,
                'model_used': model,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error in generic LLM request: {e}")
            return {
                'response': '',
                'error': str(e),
                'success': False
            }

    def _build_content_analysis_prompt(self, content: str, analysis_type: str) -> str:
        """Build prompt for content analysis."""
        if analysis_type == 'quality':
            return f"""
Analyze the quality of this article content. Rate it on a scale of 1-10 and provide reasoning:

Content: {content[:2000]}...

Provide your analysis in this format:
Quality Score: [1-10]
Reasoning: [Your detailed reasoning]
Key Strengths: [List strengths]
Areas for Improvement: [List areas]
"""
        elif analysis_type == 'relevance':
            return f"""
Analyze the relevance and topic categorization of this article:

Content: {content[:2000]}...

Provide your analysis in this format:
Primary Topic: [Main topic]
Secondary Topics: [List related topics]
Target Audience: [Intended audience]
Relevance Score: [1-10]
"""
        else:
            return f"""
Provide a comprehensive analysis of this article content:

Content: {content[:2000]}...

Analyze the content for:
1. Quality and readability
2. Main topics and themes  
3. Target audience
4. Key insights or takeaways
5. Overall assessment

Provide a structured analysis.
"""

    def _build_search_enhancement_prompt(self, query: str, context: str) -> str:
        """Build prompt for search query enhancement."""
        return f"""
Enhance this search query to make it more effective for finding relevant articles:

Original Query: {query}
Context: {context}

Provide an enhanced search query that:
1. Includes relevant synonyms and related terms
2. Uses appropriate search operators if needed
3. Maintains the original intent
4. Is optimized for semantic search

Enhanced Query: [Your enhanced query here]
"""

    def _build_summarization_prompt(self, content: str, max_length: int) -> str:
        """Build prompt for summarization."""
        return f"""
Create a concise summary of this article content in approximately {max_length} characters:

Content: {content}

Summary requirements:
- Capture the main points and key insights
- Maintain the original tone and intent
- Be informative yet concise
- Focus on actionable information if present

Summary:
"""

    def _build_classification_prompt(self, content: str, categories: List[str]) -> str:
        """Build prompt for classification."""
        categories_str = ", ".join(categories) if categories else "Technology, Business, Science, Health, Politics, Entertainment, Sports, Other"
        
        return f"""
Classify this article content into the most appropriate category:

Content: {content[:1500]}...

Available Categories: {categories_str}

Based on the content, which category best fits this article? Provide just the category name.

Category:
"""
