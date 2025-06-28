"""
Local LLM Service using Llama-2-7B-Chat for efficient content analysis.

This service replaces DistilGPT-2 with a more capable local LLM that can handle
the hybrid RAG approach with proper context windows.

Key Features:
- Llama-2-7B-Chat (quantized for 8GB RAM)
- 4096 token context window
- Apple Silicon optimized via llama.cpp
- Memory-efficient inference
"""

import os
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    logging.warning("llama-cpp-python not available - falling back to transformers")

# Fallback imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class LocalLLMService:
    """
    Local LLM service optimized for MacBook Air 8GB RAM.
    
    Uses Llama-2-7B-Chat with quantization for memory efficiency.
    Provides content analysis, relevance scoring, and synthesis capabilities.
    """
    
    def __init__(self, model_path: str = None, use_quantized: bool = True):
        self.model_path = model_path
        self.use_quantized = use_quantized
        self.model = None
        self.context_window = 4096
        self.max_tokens_per_request = 512
        
        # Model configuration for different use cases
        self.analysis_config = {
            "temperature": 0.1,  # Low temperature for analytical tasks
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1
        }
        
        self.synthesis_config = {
            "temperature": 0.3,  # Slightly higher for creative synthesis
            "top_p": 0.95,
            "top_k": 50,
            "repeat_penalty": 1.1
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the local LLM with appropriate configuration."""
        if not HAS_LLAMA_CPP:
            logger.warning("llama-cpp-python not available, falling back to transformers")
            self._initialize_fallback_model()
            return
        
        try:
            # Try to download/load Llama-2-7B-Chat model
            logger.info("Initializing Llama-2-7B-Chat model...")
            
            # Default model path - Docker and local compatible
            if not self.model_path:
                # Check for Docker environment first
                docker_model_path = os.getenv('LLM_MODEL_PATH', '/app/models/local_llm_model.gguf')
                if os.path.exists(docker_model_path):
                    self.model_path = docker_model_path
                else:
                    # Fallback to local development path
                    model_dir = os.path.join(os.getcwd(), "models")
                    os.makedirs(model_dir, exist_ok=True)
                    # Try the symlink first, then the actual file
                    symlink_path = os.path.join(model_dir, "local_llm_model.gguf")
                    direct_path = os.path.join(model_dir, "llama-2-7b-chat.q4_0.gguf")
                    if os.path.exists(symlink_path):
                        self.model_path = symlink_path
                    else:
                        self.model_path = direct_path
            
            # Check if model exists, if not, provide instructions
            if not os.path.exists(self.model_path):
                logger.error(f"Model not found at {self.model_path}")
                logger.info("Please download the model using the download script")
                self._initialize_fallback_model()
                return
            
            # Initialize with memory-efficient settings for 8GB RAM
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_window,  # Context window
                n_batch=512,  # Batch size for prompt processing
                n_threads=4,  # Use 4 threads for MacBook Air
                n_gpu_layers=0,  # Use CPU only (safer for 8GB RAM)
                verbose=False
            )
            
            logger.info("✓ Llama-2-7B-Chat model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {e}")
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize fallback model (Phi-3-mini) if Llama fails."""
        if not HAS_TRANSFORMERS:
            logger.error("No LLM available - both llama-cpp and transformers failed")
            return
        
        try:
            logger.info("Initializing fallback model: Microsoft Phi-3-mini-4k-instruct")
            self.model = pipeline(
                "text-generation",
                model="microsoft/Phi-3-mini-4k-instruct",
                device=-1,  # CPU only
                torch_dtype="auto",
                trust_remote_code=True
            )
            self.context_window = 4096  # 4K context window for Phi-3
            self.max_tokens_per_request = 512
            logger.info("✓ Phi-3-mini fallback model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Phi-3-mini, trying smaller model: {e}")
            # Final fallback to a very lightweight model if Phi-3 fails
            try:
                logger.info("Trying final fallback: TinyLlama")
                self.model = pipeline(
                    "text-generation",
                    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    device=-1,  # CPU only
                    torch_dtype="auto"
                )
                self.context_window = 2048
                logger.info("✓ TinyLlama fallback model initialized")
            except Exception as tiny_error:
                logger.error(f"All fallback models failed: {tiny_error}")
                self.model = None
    
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        return self.model is not None
    
    async def generate_response(self, prompt: str, config: Dict = None) -> str:
        """Public method to generate response from the LLM."""
        if not config:
            config = self.analysis_config
        return await self._generate_response(prompt, config)
    
    async def analyze_content_relevance(self, 
                                      query: str, 
                                      articles: List[Dict],
                                      top_k: int = 10) -> List[Dict]:
        """
        Analyze article relevance using LLM for hybrid RAG approach.
        
        Args:
            query: User search query
            articles: List of article candidates from vector search
            top_k: Number of top results to return
            
        Returns:
            List of articles with LLM relevance scores and analysis
        """
        if not self.is_available():
            logger.warning("LLM not available, returning articles without analysis")
            return articles[:top_k]
        
        try:
            logger.info(f"Analyzing {len(articles)} articles for query: '{query}'")
            
            analyzed_articles = []
            
            # Process articles in batches to fit context window
            batch_size = self._calculate_batch_size(articles)
            
            for i in range(0, len(articles), batch_size):
                batch = articles[i:i + batch_size]
                batch_results = await self._analyze_article_batch(query, batch)
                analyzed_articles.extend(batch_results)
            
            # Sort by LLM relevance score
            analyzed_articles.sort(
                key=lambda x: x.get('llm_relevance_score', 0), 
                reverse=True
            )
            
            logger.info(f"✓ Analysis complete, returning top {top_k} results")
            return analyzed_articles[:top_k]
            
        except Exception as e:
            logger.error(f"Error in content relevance analysis: {e}")
            return articles[:top_k]
    
    async def _analyze_article_batch(self, query: str, articles: List[Dict]) -> List[Dict]:
        """Analyze a batch of articles that fit in context window."""
        try:
            # Create analysis prompt
            prompt = self._create_relevance_prompt(query, articles)
            
            # Get LLM response
            response = await self._generate_response(prompt, self.analysis_config)
            
            # Parse response and enhance articles
            enhanced_articles = self._parse_relevance_response(response, articles)
            
            return enhanced_articles
            
        except Exception as e:
            logger.error(f"Error analyzing article batch: {e}")
            # Return articles with default scores if analysis fails
            for article in articles:
                article['llm_relevance_score'] = 0.5
                article['llm_analysis'] = "Analysis failed"
            return articles
    
    def _create_relevance_prompt(self, query: str, articles: List[Dict]) -> str:
        """Create a prompt for analyzing article relevance."""
        prompt = f"""
You are an expert content analyst. Analyze these articles for relevance to the user's query.

Query: "{query}"

Articles to analyze:
"""
        
        for i, article in enumerate(articles, 1):
            # Include title, summary, and first part of content
            content_preview = article.get('content', article.get('summary', ''))[:300]
            prompt += f"""
{i}. Title: {article.get('title', 'Unknown')}
   Content: {content_preview}...
"""
        
        prompt += """
For each article, provide:
1. Relevance score (0.0-1.0)
2. Brief relevance explanation
3. Key matching concepts

Format your response as:
Article 1: Score=0.8, Explanation=Directly addresses the query topic, Concepts=AI, machine learning
Article 2: Score=0.3, Explanation=Tangentially related, Concepts=technology
...

Focus on semantic relevance, not just keyword matching.
"""
        
        return prompt
    
    async def _generate_response(self, prompt: str, config: Dict) -> str:
        """Generate response from the LLM."""
        try:
            if hasattr(self.model, '__call__') and HAS_LLAMA_CPP:
                # llama.cpp model
                response = self.model(
                    prompt,
                    max_tokens=self.max_tokens_per_request,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    repeat_penalty=config["repeat_penalty"],
                    stop=["Human:", "User:", "\n\n\n"]
                )
                return response["choices"][0]["text"].strip()
            
            else:
                # Transformers fallback with truncation for safety
                prompt_length = len(prompt.split())
                
                # Truncate prompt if too long for the model
                if prompt_length > 700:  # Leave room for response
                    words = prompt.split()
                    prompt = " ".join(words[:700]) + "..."
                    self.logger.warning(f"Truncated prompt from {prompt_length} to 700 words")
                
                # Generate with conservative settings
                response = self.model(
                    prompt,
                    max_length=min(900, len(prompt.split()) + 100),  # Conservative max length
                    temperature=config["temperature"],
                    do_sample=True,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    repetition_penalty=1.1,
                    early_stopping=True
                )
                generated_text = response[0]["generated_text"]
                result = generated_text[len(prompt):].strip()
                
                # Return a minimal result if generation is too short or failed
                if len(result) < 10:
                    return "Relevance Score: 0.5\nReasoning: Analysis completed\nKey Topics: general"
                
                return result
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "Analysis failed due to generation error"
    
    def _parse_relevance_response(self, response: str, articles: List[Dict]) -> List[Dict]:
        """Parse LLM response and enhance articles with analysis."""
        enhanced_articles = []
        
        try:
            lines = response.split('\n')
            
            for i, article in enumerate(articles):
                # Find the line for this article
                article_line = None
                for line in lines:
                    if line.strip().startswith(f"Article {i+1}:"):
                        article_line = line
                        break
                
                if article_line:
                    # Parse score, explanation, and concepts
                    score = self._extract_score(article_line)
                    explanation = self._extract_explanation(article_line)
                    concepts = self._extract_concepts(article_line)
                else:
                    # Default values if parsing fails
                    score = 0.5
                    explanation = "Analysis incomplete"
                    concepts = []
                
                # Enhance article with LLM analysis
                enhanced_article = {
                    **article,
                    'llm_relevance_score': score,
                    'llm_explanation': explanation,
                    'llm_concepts': concepts,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                enhanced_articles.append(enhanced_article)
        
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            # Return articles with default analysis
            for article in articles:
                enhanced_articles.append({
                    **article,
                    'llm_relevance_score': 0.5,
                    'llm_explanation': "Parsing failed",
                    'llm_concepts': []
                })
        
        return enhanced_articles
    
    def _extract_score(self, line: str) -> float:
        """Extract relevance score from analysis line."""
        try:
            import re
            score_match = re.search(r'Score=([0-9.]+)', line)
            if score_match:
                return float(score_match.group(1))
        except:
            pass
        return 0.5
    
    def _extract_explanation(self, line: str) -> str:
        """Extract explanation from analysis line."""
        try:
            import re
            exp_match = re.search(r'Explanation=([^,]+)', line)
            if exp_match:
                return exp_match.group(1).strip()
        except:
            pass
        return "No explanation available"
    
    def _extract_concepts(self, line: str) -> List[str]:
        """Extract key concepts from analysis line."""
        try:
            import re
            concepts_match = re.search(r'Concepts=(.+)', line)
            if concepts_match:
                concepts_str = concepts_match.group(1).strip()
                return [c.strip() for c in concepts_str.split(',')]
        except:
            pass
        return []
    
    def _calculate_batch_size(self, articles: List[Dict]) -> int:
        """Calculate optimal batch size based on context window."""
        # Estimate tokens per article (title + content preview)
        avg_tokens_per_article = 100  # Conservative estimate
        prompt_overhead = 200  # Tokens for prompt structure
        response_tokens = 300  # Tokens reserved for response
        
        available_tokens = self.context_window - prompt_overhead - response_tokens
        max_articles = max(1, available_tokens // avg_tokens_per_article)
        
        # Limit to reasonable batch size
        return min(max_articles, 10)
    
    async def synthesize_search_results(self, query: str, articles: List[Dict]) -> str:
        """Synthesize search results into a coherent summary."""
        if not self.is_available():
            return f"Found {len(articles)} articles related to '{query}'"
        
        try:
            # Create synthesis prompt
            prompt = f"""
Synthesize these search results into a coherent summary for the user.

Query: "{query}"

Top relevant articles:
"""
            
            for i, article in enumerate(articles[:5], 1):  # Top 5 for synthesis
                prompt += f"""
{i}. {article.get('title', 'Unknown')}
   Relevance: {article.get('llm_relevance_score', 0.5):.2f}
   Summary: {article.get('summary', '')[:200]}...
"""
            
            prompt += """
Provide a concise synthesis that:
1. Summarizes the key themes across these articles
2. Highlights the most relevant findings for the query
3. Mentions specific article insights where appropriate

Keep it under 200 words.
"""
            
            response = await self._generate_response(prompt, self.synthesis_config)
            return response
            
        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            return f"Found {len(articles)} articles related to '{query}'"


# Global service instance
local_llm_service = LocalLLMService()
