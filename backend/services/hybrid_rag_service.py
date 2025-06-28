"""
Hybrid RAG Search Service

Combines vector similarity search with LLM-powered contextualization and analysis.
This service provides:
1. Vector-based semantic search for initial retrieval
2. LLM analysis for relevance scoring and context synthesis
3. Ranked and enhanced results with AI insights

Architecture:
- Pre-filtering with vector search (top 50-100 candidates)
- LLM scoring and analysis of candidates
- Final ranking and result synthesis
"""

import logging
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
from backend.config import config
from backend.services.ollama_service import ollama_service
from backend.monitoring import (
    monitor_database_operation,
    monitor_llm_operation,
    monitor_agent_operation,
    record_llm_tokens
)
from backend.services.rag_evaluation_service import rag_evaluation_service

class HybridRAGService:
    """Enhanced search service combining vector search with LLM analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger("HybridRAGService")
        self.embedding_model = None
        self.llm_service = ollama_service
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model."""
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"Initialized embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
    
    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASS,
            database=config.DB_NAME,
            cursor_factory=RealDictCursor
        )
    
    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for search query."""
        try:
            if not self.embedding_model:
                self.logger.error("Embedding model not initialized")
                return None
            
            embedding = self.embedding_model.encode(query)
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            return None
    
    async def search_and_analyze(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Perform multi-strategy search with fallback mechanisms.
        
        Args:
            query: Search query
            top_k: Final number of results to return
            
        Returns:
            Dict with results using best available search strategy
        """
        try:
            self.logger.info(f"Starting multi-strategy search for: {query}")
            
            # Strategy 1: Try vector search with original query
            self.logger.info("Attempting vector search")
            search_results = await self._vector_search_with_llm_summaries(query, top_k)
            search_method = "vector_original"
            
            # Strategy 2: If no results, try individual terms (query expansion)
            if not search_results:
                self.logger.info(f"No results for '{query}', trying individual terms...")
                self.logger.info("Attempting individual terms search")
                search_results = await self._search_individual_terms(query, top_k)
                search_method = "individual_terms"
                
            # Strategy 3: If still no results, try keyword-based search
            if not search_results:
                self.logger.info(f"No results for individual terms, trying keyword search...")
                self.logger.info("Attempting keyword search")
                search_results = await self._keyword_search(query, top_k)
                search_method = "keyword_search"
                
            # Strategy 4: If still no results, try fuzzy/partial matching
            if not search_results:
                self.logger.info(f"No results for keyword search, trying fuzzy search...")
                self.logger.info("Attempting fuzzy search")
                search_results = await self._fuzzy_search(query, top_k)
                search_method = "fuzzy_search"
                
            if not search_results:
                return {
                    "results": [], 
                    "total_found": 0, 
                    "query": query,
                    "search_method": "all_strategies_failed",
                    "message": "No relevant articles found with any search strategy"
                }
            
            # Optional: Generate context for good results
            context = None
            if len(search_results) >= 3:
                try:
                    import asyncio
                    context = await asyncio.wait_for(
                        self._generate_query_context(query, search_results[:3]),
                        timeout=20.0
                    )
                except Exception as e:
                    self.logger.warning(f"Context generation failed: {e}")
            
            self.logger.info(f"Multi-strategy search completed: {len(search_results)} results using {search_method}")
            
            # Build response
            response = {
                "results": search_results,
                "total_found": len(search_results),
                "query": query,
                "search_method": search_method,
                "llm_enhanced": True
            }
            
            if context:
                response["context"] = context
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error in multi-strategy search: {e}")
            return {"results": [], "total_found": 0, "error": str(e)}
    
    async def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform initial vector similarity search."""
        with monitor_database_operation("vector_search", config.VECTOR_TABLE_NAME):
            try:
                # Generate query embedding
                query_embedding = self.generate_query_embedding(query)
                if not query_embedding:
                    return []
                
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Vector similarity search with proper type casting
                cursor.execute(f"""
                    SELECT 
                        title, link, summary, content, author, hash, processed_at, digest_date,
                        (embedding <=> %s::vector) as distance
                    FROM {config.VECTOR_TABLE_NAME}
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, limit))
                
                results = cursor.fetchall()
                
                # Format results
                candidates = []
                for result in results:
                    article = dict(result)
                    # Convert distance to similarity score
                    similarity = max(0, 1 - result['distance']) if result['distance'] is not None else 0
                    article['vector_score'] = similarity
                    candidates.append(article)
                
                cursor.close()
                conn.close()
                
                self.logger.info(f"Vector search found {len(candidates)} candidates")
                return candidates
                
            except Exception as e:
                self.logger.error(f"Error in vector search: {e}")
                return []
    
    async def _llm_analyze_and_rank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Use LLM to analyze and re-rank search candidates with context generation."""
        try:
            self.logger.info(f"LLM analyzing {len(candidates)} candidates for query: {query}")
            
            # First, generate overall context from top candidates
            context = await self._generate_search_context(query, candidates[:5])
            
            enhanced_results = []
            
            for i, article in enumerate(candidates[:top_k]):
                # Clean up summary - ensure consistent 2-3 sentence format
                cleaned_summary = await self._ensure_consistent_summary(article)
                article['summary'] = cleaned_summary
                
                # Add relevance score (use vector score as base)
                article['score'] = article.get('vector_score', 0.5)
                article['query_matched'] = query
                article['analysis_method'] = 'hybrid_rag_enhanced'
                
                enhanced_results.append(article)
            
            # Add context to results
            self.logger.info(f"Generated context and enhanced {len(enhanced_results)} results")
            return enhanced_results, context
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {e}")
            # Fallback to original results
            fallback_results = []
            for article in candidates[:top_k]:
                article.update({
                    'score': article.get('vector_score', 0.5),
                    'query_matched': query,
                    'analysis_method': 'vector_fallback'
                })
                fallback_results.append(article)
            return fallback_results, None
    
    async def _ensure_consistent_summary(self, article: Dict[str, Any]) -> str:
        """Ensure article summary is consistent 2-3 sentences."""
        try:
            current_summary = article.get('summary', '')
            title = article.get('title', 'Unknown Title')
            
            # If summary is already good (2-3 sentences, reasonable length)
            sentence_count = current_summary.count('.') + current_summary.count('!') + current_summary.count('?')
            if 50 <= len(current_summary) <= 300 and 2 <= sentence_count <= 4:
                return current_summary
            
            # Otherwise, generate a consistent summary
            content_preview = article.get('content', current_summary)[:1000]
            
            summary_prompt = f"""
Create a concise 2-3 sentence summary of this article. Focus on key insights and practical value.

Title: {title}
Content: {content_preview}

Summary:"""
            
            # Generate with longer timeout and no rate limiting for search
            import asyncio
            generated = await asyncio.wait_for(
                ollama_service.generate_response(summary_prompt),
                timeout=15.0  # Increased timeout
            )
            
            if generated and len(generated.strip()) > 30:
                clean_summary = generated.strip()
                # Remove "Summary:" prefix if present
                if clean_summary.lower().startswith('summary:'):
                    clean_summary = clean_summary[8:].strip()
                return clean_summary
            else:
                return current_summary  # Fallback to original
                
        except Exception as e:
            self.logger.warning(f"Failed to generate consistent summary: {e}")
            return article.get('summary', 'Summary not available')
    
    async def _generate_search_context(self, query: str, top_articles: List[Dict[str, Any]]) -> str:
        """Generate contextual summary for search results."""
        try:
            if not top_articles:
                return None
            
            # Prepare articles for context generation
            articles_text = "\n\n".join([
                f"Article {i+1}: {article.get('title', 'N/A')}\n{article.get('summary', 'No summary')[:200]}..."
                for i, article in enumerate(top_articles[:3])
            ])
            
            context_prompt = f"""
Based on these search results for "{query}", provide a brief 2-3 sentence overview of the key themes and insights.

{articles_text}

Context:"""
            
            import asyncio
            context = await asyncio.wait_for(
                ollama_service.generate_response(context_prompt),
                timeout=15.0  # Increased timeout
            )
            
            if context and len(context.strip()) > 20:
                return context.strip()
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to generate search context: {e}")
            return None
    
    async def _vector_search_with_llm_summaries(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform vector search and return results with pre-computed LLM summaries."""
        with monitor_database_operation("vector_search_llm", config.VECTOR_TABLE_NAME):
            try:
                # Generate query embedding
                query_embedding = self.generate_query_embedding(query)
                if not query_embedding:
                    return []
                
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Vector similarity search with LLM summaries - get extra results for filtering
                search_limit = limit * 3  # Get 3x more results for deduplication and filtering
                cursor.execute(f"""
                    SELECT 
                        title, link, summary, llm_summary, content, author, hash, processed_at, digest_date,
                        (embedding <=> %s::vector) as distance
                    FROM {config.VECTOR_TABLE_NAME}
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, search_limit))
                
                results = cursor.fetchall()
                
                # Format and filter results
                search_results = []
                seen_content_hashes = set()
                seen_titles = set()
                
                for result in results:
                    article = dict(result)
                    
                    # Convert distance to similarity score
                    similarity = max(0, 1 - result['distance']) if result['distance'] is not None else 0
                    article['score'] = similarity
                    article['vector_score'] = similarity
                    
                    # Apply adaptive relevance filtering
                    min_threshold = self._calculate_adaptive_threshold(query, similarity, search_results)
                    if similarity < min_threshold:
                        continue
                    
                    # Skip publication pages and generic content
                    title = article.get('title', '').lower()
                    if any(generic in title for generic in [
                        'data science collective',
                        'artificial intelligence in plain english',
                        'tech leadership unlocked',
                        'control your recommendations',
                        'switch to the weekly digest',
                        'the medium newsletter'
                    ]):
                        continue
                    
                    # Deduplicate by content hash and similar titles
                    content_hash = article.get('hash', '')
                    clean_title = self._clean_title_for_comparison(article.get('title', ''))
                    
                    if content_hash in seen_content_hashes or clean_title in seen_titles:
                        continue
                    
                    seen_content_hashes.add(content_hash)
                    seen_titles.add(clean_title)
                    
                    # Use LLM summary if available, otherwise fall back to original summary
                    if article.get('llm_summary') and len(article['llm_summary'].strip()) > 20:
                        article['summary'] = article['llm_summary']
                        article['summary_type'] = 'llm_generated'
                    else:
                        article['summary_type'] = 'original'
                    
                    # Skip articles with poor summaries
                    summary = article.get('summary', '')
                    if summary.startswith('Summary for:') and len(summary) < 100:
                        continue
                    
                    # Add metadata
                    article['query_matched'] = query
                    article['analysis_method'] = 'vector_with_precomputed_llm'
                    
                    search_results.append(article)
                    
                    # Stop when we have enough good results
                    if len(search_results) >= limit:
                        break
                
                cursor.close()
                conn.close()
                
                self.logger.info(f"Vector search found {len(search_results)} unique, relevant results after filtering")
                return search_results
                
            except Exception as e:
                self.logger.error(f"Error in vector search with LLM summaries: {e}")
                return []
    
    async def _generate_query_context(self, query: str, top_results: List[Dict[str, Any]]) -> str:
        """Generate query-specific contextual overview from search results."""
        try:
            if not top_results:
                return None
            
            # Prepare top results for context generation
            results_text = "\n\n".join([
                f"Result {i+1}: {article.get('title', 'N/A')}\n{article.get('summary', 'No summary')[:300]}..."
                for i, article in enumerate(top_results[:3])
            ])
            
            context_prompt = f"""
Based on these search results for the query "{query}", provide a brief 2-3 sentence contextual overview highlighting the key themes, insights, and relevance to the search query.

Top Search Results:
{results_text}

Contextual Overview:"""
            
            # Generate context with timeout
            context = await ollama_service.generate_response(context_prompt)
            
            if context and len(context.strip()) > 30:
                # Clean the response
                clean_context = context.strip()
                if clean_context.lower().startswith('contextual overview:'):
                    clean_context = clean_context[20:].strip()
                return clean_context
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to generate query context: {e}")
            return None
    
    def _calculate_adaptive_threshold(self, query: str, current_similarity: float, current_results: List[Dict[str, Any]]) -> float:
        """Calculate adaptive relevance threshold based on query characteristics and result quality."""
        
        # Base threshold - never go below 5% to filter out truly irrelevant content
        base_threshold = 0.05
        
        # Query characteristics
        query_words = len(query.split())
        has_specific_terms = any(term in query.lower() for term in [
            'claude', 'anthropic', 'gpt', 'openai', 'python', 'javascript', 'react', 'vue', 'angular'
        ])
        
        # Special handling for single high-value terms
        is_single_ai_term = query_words == 1 and query.lower() in ['claude', 'anthropic', 'gpt', 'openai', 'chatgpt']
        
        # Special case for single AI terms - be very permissive regardless of existing results
        if is_single_ai_term:
            return max(0.05, base_threshold)  # 5% threshold for single AI terms
        
        # Result quality analysis (only for non-single-AI-term queries)
        if current_results:
            scores = [r.get('score', 0) for r in current_results]
            max_score = max(scores) if scores else 0
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # If we have high-quality results, be more selective
            if max_score > 0.40:  # Very relevant content exists
                return max(0.25, base_threshold)  # 25% threshold
            elif max_score > 0.30:  # Good content exists
                return max(0.20, base_threshold)  # 20% threshold
            elif len(current_results) >= 3:  # We have some results
                return max(0.15, base_threshold)  # 15% threshold
        
        # Query-based thresholds
        if has_specific_terms and query_words <= 3:
            # Specific, focused queries - be more permissive to find relevant content
            return max(0.10, base_threshold)  # 10% threshold
        elif query_words >= 5:
            # Complex queries - use moderate threshold
            return max(0.15, base_threshold)  # 15% threshold
        else:
            # General queries - use higher threshold
            return max(0.20, base_threshold)  # 20% threshold
    
    def _clean_title_for_comparison(self, title: str) -> str:
        """Clean title for deduplication comparison."""
        import re
        # Remove tracking parameters, reading time info, and normalize
        cleaned = re.sub(r'\d+\s*min\s*read\d*', '', title.lower())
        cleaned = re.sub(r'\s+\d+\s*$', '', cleaned)
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned)
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace
        return cleaned[:100]  # Limit length for comparison
    
    async def _search_individual_terms(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for individual terms when combined query fails."""
        try:
            # Split query into individual meaningful terms
            terms = [term.strip() for term in query.split() if len(term.strip()) > 2]
            self.logger.info(f"Searching individual terms: {terms}")
            
            all_results = []
            seen_hashes = set()
            
            # Search for each term individually
            for term in terms:
                term_results = await self._vector_search_with_llm_summaries(term, limit)
                
                # Add unique results
                for result in term_results:
                    result_hash = result.get('hash', '')
                    if result_hash not in seen_hashes:
                        result['matched_term'] = term
                        all_results.append(result)
                        seen_hashes.add(result_hash)
                
                # Stop if we have enough results
                if len(all_results) >= limit:
                    break
            
            # Sort by relevance score
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in individual terms search: {e}")
            return []
    
    async def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search using PostgreSQL text search."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Use PostgreSQL full-text search
            search_terms = [term.strip() for term in query.lower().split() if len(term.strip()) > 1]
            
            if not search_terms:
                return []
            
            # Create search conditions for title and content
            search_conditions = []
            search_params = []
            
            # For multi-term queries, try AND logic first
            if len(search_terms) > 1:
                # AND logic: all terms must be present
                and_conditions = []
                for term in search_terms:
                    and_conditions.append("(LOWER(title) LIKE %s OR LOWER(content) LIKE %s OR LOWER(summary) LIKE %s)")
                    search_params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
                
                where_clause = " AND ".join(and_conditions)
            else:
                # Single term: use OR logic across fields
                term = search_terms[0]
                where_clause = "(LOWER(title) LIKE %s OR LOWER(content) LIKE %s OR LOWER(summary) LIKE %s)"
                search_params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
            
            cursor.execute(f"""
                SELECT title, link, summary, llm_summary, content, author, hash, processed_at, digest_date
                FROM {config.VECTOR_TABLE_NAME}
                WHERE {where_clause}
                ORDER BY 
                    CASE WHEN LOWER(title) LIKE %s THEN 1 ELSE 2 END,
                    LENGTH(title)
                LIMIT %s
            """, search_params + [f"%{search_terms[0]}%", limit])
            
            results = cursor.fetchall()
            
            # Format results
            search_results = []
            for result in results:
                article = dict(result)
                
                # Calculate relevance based on term matches
                title_lower = article.get('title', '').lower()
                content_lower = article.get('content', '').lower()
                
                # Simple relevance scoring
                relevance = 0
                for term in search_terms:
                    if term in title_lower:
                        relevance += 0.3  # Title matches are more important
                    if term in content_lower:
                        relevance += 0.1
                
                article['score'] = min(relevance, 0.9)  # Cap at 90%
                article['vector_score'] = article['score']
                article['analysis_method'] = 'keyword_search'
                
                # Use LLM summary if available
                if article.get('llm_summary') and len(article['llm_summary'].strip()) > 20:
                    article['summary'] = article['llm_summary']
                    article['summary_type'] = 'llm_generated'
                else:
                    article['summary_type'] = 'original'
                
                search_results.append(article)
            
            cursor.close()
            conn.close()
            
            # Sort by relevance
            search_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            self.logger.info(f"Keyword search found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}")
            return []
    
    async def _fuzzy_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform fuzzy/partial matching search."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Use PostgreSQL similarity and partial matching
            query_lower = query.lower()
            
            # Fuzzy search using ILIKE with wildcards
            cursor.execute(f"""
                SELECT title, link, summary, llm_summary, content, author, hash, processed_at, digest_date,
                       similarity(LOWER(title), %s) as title_similarity
                FROM {config.VECTOR_TABLE_NAME}
                WHERE 
                    similarity(LOWER(title), %s) > 0.1 OR
                    LOWER(title) ILIKE %s OR
                    LOWER(content) ILIKE %s
                ORDER BY 
                    similarity(LOWER(title), %s) DESC,
                    LENGTH(title)
                LIMIT %s
            """, (query_lower, query_lower, f"%{query_lower}%", f"%{query_lower}%", query_lower, limit))
            
            results = cursor.fetchall()
            
            # Format results
            search_results = []
            for result in results:
                article = dict(result)
                
                # Use title similarity as base score
                similarity_score = result.get('title_similarity', 0)
                article['score'] = max(similarity_score, 0.15)  # Minimum 15% for fuzzy matches
                article['vector_score'] = article['score']
                article['analysis_method'] = 'fuzzy_search'
                
                # Use LLM summary if available
                if article.get('llm_summary') and len(article['llm_summary'].strip()) > 20:
                    article['summary'] = article['llm_summary']
                    article['summary_type'] = 'llm_generated'
                else:
                    article['summary_type'] = 'original'
                
                search_results.append(article)
            
            cursor.close()
            conn.close()
            
            self.logger.info(f"Fuzzy search found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in fuzzy search: {e}")
            return []
    
    def get_article_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Get total articles
            cursor.execute(f"SELECT COUNT(*) as total FROM {config.VECTOR_TABLE_NAME}")
            total_result = cursor.fetchone()
            total_articles = total_result['total'] if total_result else 0
            
            # Get date range
            cursor.execute(f"""
                SELECT 
                    MIN(digest_date) as earliest_digest,
                    MAX(digest_date) as latest_digest,
                    COUNT(DISTINCT digest_date) as total_digest_days
                FROM {config.VECTOR_TABLE_NAME}
                WHERE digest_date IS NOT NULL
            """)
            date_result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            stats = {
                "total_articles": total_articles,
                "database_info": {
                    "database_name": config.DB_NAME,
                    "table_name": config.VECTOR_TABLE_NAME,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }
            
            if date_result:
                stats.update({
                    "earliest_digest": date_result['earliest_digest'].isoformat() if date_result['earliest_digest'] else None,
                    "latest_digest": date_result['latest_digest'].isoformat() if date_result['latest_digest'] else None,
                    "total_digest_days": date_result['total_digest_days'] or 0
                })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting article stats: {e}")
            return {
                "total_articles": 0,
                "error": str(e),
                "database_info": {
                    "database_name": config.DB_NAME,
                    "table_name": config.VECTOR_TABLE_NAME,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }

# Global service instance
hybrid_rag_service = HybridRAGService()
