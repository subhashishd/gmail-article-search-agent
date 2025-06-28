"""
Ollama Service for Local LLM Operations

Uses Ollama to interface with local LLM models (Llama 3.2 1B) for:
- Content analysis and relevance scoring
- RAG-based response generation
- Efficient inference on limited hardware
"""

import asyncio
import httpx
import json
import logging
from typing import Dict, List, Optional, Any
from backend.config import config
from backend.monitoring import monitor_llm_operation, record_llm_tokens

logger = logging.getLogger(__name__)

class OllamaService:
    """Service for interacting with Ollama local LLM models."""
    
    def __init__(self):
        self.base_url = config.OLLAMA_HOST if hasattr(config, 'OLLAMA_HOST') else "http://ollama:11434"
        self.model_name = config.OLLAMA_MODEL if hasattr(config, 'OLLAMA_MODEL') else "llama3.2:1b"
        self.max_tokens = 512
        self.temperature = 0.1  # Low temperature for analytical tasks
        
        logger.info(f"Initializing Ollama service with {self.model_name} at {self.base_url}")
    
    async def is_available(self) -> bool:
        """Check if Ollama service and model are available."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check if Ollama is running
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    # Check if our model is available
                    model_available = any(
                        model['name'] == self.model_name 
                        for model in models
                    )
                    if model_available:
                        logger.info(f"✓ {self.model_name} is available in Ollama")
                        return True
                    else:
                        logger.warning(f"Model {self.model_name} not found in Ollama")
                        return False
                return False
        except Exception as e:
            logger.error(f"Ollama service check failed: {e}")
            return False
    
    async def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using Ollama."""
        try:
            with monitor_llm_operation("ollama", self.model_name, "generate"):
                async with httpx.AsyncClient(timeout=60.0) as client:
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                            "top_p": 0.9,
                            "repeat_penalty": 1.1
                        }
                    }
                    
                    if system_prompt:
                        payload["system"] = system_prompt
                    
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        generated_text = result.get('response', '')
                        
                        # Record token usage (estimates)
                        input_tokens = len(prompt.split()) * 1.3
                        output_tokens = len(generated_text.split()) * 1.3
                        record_llm_tokens("ollama", self.model_name, int(input_tokens), int(output_tokens))
                        
                        return generated_text.strip()
                    else:
                        logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                        return "Error: Unable to generate response"
                        
        except asyncio.TimeoutError:
            logger.error("Ollama request timed out")
            return "Error: Request timed out"
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            return f"Error: {str(e)}"
    
    async def analyze_relevance(self, query: str, title: str, content: str) -> Dict[str, Any]:
        """Analyze article relevance for a given query."""
        
        # Truncate content to fit context window
        content_preview = content[:1000] if len(content) > 1000 else content
        
        system_prompt = """You are an expert content analyst. Analyze the relevance of articles to search queries.
Provide precise, numerical relevance scores and clear reasoning."""
        
        user_prompt = f"""Analyze the relevance of this article to the search query.

Query: "{query}"

Article Title: {title}
Article Content: {content_preview}

Rate the relevance on a scale of 0.0 to 1.0 and provide reasoning.
Consider:
1. Topic alignment with the query
2. Depth of coverage
3. Practical value
4. Content quality

Respond EXACTLY in this format:
Relevance Score: [0.0-1.0]
Reasoning: [Brief explanation]
Key Topics: [Comma-separated list of main topics]"""

        try:
            response = await self.generate_response(user_prompt, system_prompt)
            return self._parse_relevance_response(response)
        except Exception as e:
            logger.error(f"Error in relevance analysis: {e}")
            return {
                'relevance_score': 0.5,
                'reasoning': 'Analysis failed',
                'key_topics': []
            }
    
    def _parse_relevance_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM relevance analysis response."""
        try:
            lines = response.strip().split('\n')
            
            relevance_score = 0.5  # Default
            reasoning = "Analysis unavailable"
            key_topics = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('Relevance Score:'):
                    try:
                        score_text = line.split(':', 1)[1].strip()
                        relevance_score = float(score_text)
                        relevance_score = max(0.0, min(1.0, relevance_score))  # Clamp to [0,1]
                    except (ValueError, IndexError):
                        pass
                elif line.startswith('Reasoning:'):
                    try:
                        reasoning = line.split(':', 1)[1].strip()
                    except IndexError:
                        pass
                elif line.startswith('Key Topics:'):
                    try:
                        topics_text = line.split(':', 1)[1].strip()
                        key_topics = [topic.strip() for topic in topics_text.split(',') if topic.strip()]
                    except IndexError:
                        pass
            
            return {
                'relevance_score': relevance_score,
                'reasoning': reasoning,
                'key_topics': key_topics
            }
            
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")
            return {
                'relevance_score': 0.5,
                'reasoning': 'Parse error',
                'key_topics': []
            }
    
    async def generate_summary(self, query: str, articles: List[Dict[str, Any]]) -> str:
        """Generate a summary answer based on retrieved articles."""
        if not articles:
            return "No relevant articles found."
        
        # Use top 3 articles for summary
        top_articles = articles[:3]
        
        # Create context from articles
        contexts = []
        for i, article in enumerate(top_articles, 1):
            title = article.get('title', 'Unknown')
            content = article.get('content', article.get('summary', ''))[:500]
            contexts.append(f"Article {i}: {title}\n{content}")
        
        combined_context = "\n\n".join(contexts)
        
        system_prompt = """You are a helpful research assistant. Provide comprehensive, accurate answers based on the given articles.
Focus on being informative and well-structured."""
        
        user_prompt = f"""Based on the following articles, provide a comprehensive answer to this question: "{query}"

Articles:
{combined_context}

Provide a well-structured answer that synthesizes information from these articles. Be specific and cite relevant points."""
        
        try:
            response = await self.generate_response(user_prompt, system_prompt)
            return response if response else "Unable to generate summary."
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary."

# Global service instance
ollama_service = OllamaService()
