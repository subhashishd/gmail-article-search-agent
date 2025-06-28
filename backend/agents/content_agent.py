"""
Content Agent for processing articles with parallel workers and rate limiting.
Listens to article.discovered events and processes them with content fetching and analysis.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
from backend.core.event_bus import event_bus
from backend.core.rate_limiter import RateLimitedRequest, rate_limiter
from backend.services.content_extractor import fetch_article_content
from backend.services.ollama_service import ollama_service

class ContentAgent:
    """Agent for processing article content in parallel with rate limiting"""
    
    def __init__(self, agent_id: str, max_workers: int = 5):
        self.agent_id = agent_id
        self.max_workers = max_workers
        self.logger = logging.getLogger(f"ContentAgent-{self.agent_id}")
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        self.processing_queue = asyncio.Queue()
        self.workers = []
        self.running = False
    
    async def initialize(self):
        """Initialize Content Agent resources"""
        self.logger.info(f"Initializing Content Agent with {self.max_workers} workers...")
        
        # Subscribe to article discovery events
        await event_bus.subscribe("article.discovered", self._handle_article_discovered)
        
        # Start worker pool
        await self._start_workers()
        self.running = True
        
    async def _start_workers(self):
        """Start parallel worker pool for processing articles"""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker_{i}"))
            self.workers.append(worker)
            self.logger.info(f"Started worker_{i}")
    
    async def _worker(self, worker_name: str):
        """Individual worker for processing articles"""
        self.logger.info(f"{worker_name} started")
        
        while self.running:
            try:
                # Get article from queue with timeout
                article = await asyncio.wait_for(
                    self.processing_queue.get(), 
                    timeout=5.0
                )
                
                await self._process_article(article, worker_name)
                
            except asyncio.TimeoutError:
                # No articles to process, continue waiting
                continue
            except Exception as e:
                self.logger.error(f"{worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _handle_article_discovered(self, event):
        """Handle article discovery events"""
        article = event.data
        self.logger.info(f"Queuing article for processing: {article.get('title', 'Unknown')}")
        
        # Add correlation ID for tracking
        article['correlation_id'] = event.id
        article['discovered_at'] = event.timestamp.isoformat()
        
        await self.processing_queue.put(article)
    
    async def _process_article(self, article: Dict[str, Any], worker_name: str):
        """Process a single article with content fetching and storage (NO LLM analysis)"""
        start_time = datetime.now()
        article_title = article.get('title', 'Unknown')
        
        try:
            self.logger.info(f"{worker_name}: Processing '{article_title}'")
            
            # Step 0: Normalize URL and check for substantial content
            article['normalized_url'] = self._normalize_url(article.get('link', ''))
            
            if not self._is_substantial_content(article):
                self.logger.info(f"{worker_name}: Skipping non-substantial content: '{article_title}'")
                return
            
            # Step 1: Fetch full content with rate limiting
            content = await self._fetch_content_with_rate_limit(article)
            article['full_content'] = content
            
            # Step 2: Generate contextual summary using LLM
            await self._generate_llm_summary(article)
            
            # Step 3: Generate embeddings and store in database
            await self._store_article_with_embeddings(article)
            
            # Step 3: Publish stored article event
            processing_time = (datetime.now() - start_time).total_seconds()
            article['processing_time_seconds'] = processing_time
            article['processed_by'] = worker_name
            article['processed_at'] = datetime.now().isoformat()
            
            await event_bus.publish(
                event_type="article.stored",
                data=article,
                source=self.agent_id,
                correlation_id=article.get('correlation_id')
            )
            
            self.logger.info(f"{worker_name}: Stored '{article_title}' in {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"{worker_name}: Failed to process '{article_title}': {e}")
            
            # Publish failed processing event for monitoring
            await event_bus.publish(
                event_type="article.processing.failed",
                data={
                    "article": article,
                    "error": str(e),
                    "worker": worker_name,
                    "failed_at": datetime.now().isoformat()
                },
                source=self.agent_id,
                correlation_id=article.get('correlation_id')
            )
    
    async def _fetch_content_with_rate_limit(self, article: Dict[str, Any]) -> str:
        """Fetch article content with rate limiting"""
        url = article.get('link', '')
        
        try:
            async with RateLimitedRequest(rate_limiter, "medium", timeout=10.0):
                content = await fetch_article_content(url)
                return content
                
        except Exception as e:
            self.logger.warning(f"Failed to fetch content for {url}: {e}")
            return article.get('summary', '')  # Fallback to summary
    
    async def _generate_llm_summary(self, article: Dict[str, Any]):
        """Generate contextual summary using LLM with improved content quality checks"""
        title = article.get('title', '')
        full_content = article.get('full_content', '')
        original_summary = article.get('summary', '')
        
        try:
            # Determine content quality and decide on summarization approach
            content_quality = self._assess_content_quality(full_content, original_summary)
            
            if content_quality['skip_llm']:
                self.logger.info(f"Skipping LLM summarization due to {content_quality['reason']}: {title[:50]}...")
                # Mark the article with content quality issues
                article['content_quality'] = content_quality['reason']
                return
            
            text_to_summarize = content_quality['text']
            is_full_content = content_quality['is_full_content']
            
            # Create enhanced summarization prompt based on content type
            if is_full_content:
                summarization_prompt = f"""
Please create a comprehensive yet concise summary of this Medium article in 2-3 sentences.
Focus on the main insights, key takeaways, and practical value for readers.

Article Title: {title}

Full Content: {text_to_summarize[:2000]}{'...' if len(text_to_summarize) > 2000 else ''}

Provide a summary that captures the article's core message:"""
            else:
                summarization_prompt = f"""
Based on this article title and brief description, provide a contextual summary in 2-3 sentences.
Note: This is based on limited content, so focus on what can be inferred about the article's likely value and topic.

Article Title: {title}

Brief Description: {text_to_summarize}

Provide a contextual summary (noting this is based on limited content):"""
            
            # Generate summary with progressive timeout strategy and content chunking
            try:
                import asyncio
                
                # Use adaptive timeout based on content size
                content_size = len(text_to_summarize)
                if content_size > 5000:  # Very large content
                    timeout = 90.0
                    # Chunk large content for better processing
                    text_to_summarize = text_to_summarize[:3000] + "..."
                    summarization_prompt = summarization_prompt.replace(text_to_summarize[:2000], text_to_summarize[:3000])
                elif content_size > 2000:  # Large content
                    timeout = 60.0
                elif is_full_content:
                    timeout = 45.0
                else:
                    timeout = 30.0
                
                self.logger.debug(f"LLM processing: {content_size} chars, timeout: {timeout}s")
                
                generated_summary = await asyncio.wait_for(
                    ollama_service.generate_response(summarization_prompt),
                    timeout=timeout
                )
                
                # Clean and validate the generated summary
                if generated_summary and len(generated_summary.strip()) > 30:
                    # Remove any "Summary:" prefix that might be included
                    cleaned_summary = generated_summary.strip()
                    if cleaned_summary.lower().startswith('summary:'):
                        cleaned_summary = cleaned_summary[8:].strip()
                    
                    # Store the LLM-generated summary with metadata
                    article['llm_summary'] = cleaned_summary
                    article['summary_type'] = 'full_content' if is_full_content else 'limited_content'
                    article['content_source'] = 'full_article' if is_full_content else 'email_summary'
                    
                    summary_length = len(cleaned_summary)
                    self.logger.info(f"Generated LLM summary ({summary_length} chars, {article['summary_type']}) for: {title[:50]}...")
                else:
                    self.logger.warning(f"LLM generated empty/short summary for: {title[:50]}...")
                    article['summary_type'] = 'failed'
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"LLM summarization timed out ({timeout}s) for: {title[:50]}...")
                article['summary_type'] = 'timeout'
            except Exception as llm_error:
                self.logger.warning(f"LLM summarization failed for {title[:50]}...: {llm_error}")
                article['summary_type'] = 'error'
                
        except Exception as e:
            self.logger.error(f"Error in LLM summarization for {title[:50]}...: {e}")
            article['summary_type'] = 'error'
    
    def _assess_content_quality(self, full_content: str, original_summary: str) -> dict:
        """Assess content quality and determine summarization strategy"""
        
        # Check if full content was successfully fetched
        if full_content and not full_content.startswith('Unable to fetch'):
            word_count = len(full_content.split())
            if word_count >= 100:  # Good quality full content
                return {
                    'text': full_content,
                    'is_full_content': True,
                    'skip_llm': False,
                    'reason': 'full_content_available'
                }
            elif word_count >= 50:  # Moderate quality content
                return {
                    'text': full_content,
                    'is_full_content': True,
                    'skip_llm': False,
                    'reason': 'partial_content_available'
                }
        
        # Fall back to original summary if available
        if original_summary and len(original_summary.strip()) >= 50:
            return {
                'text': original_summary,
                'is_full_content': False,
                'skip_llm': False,
                'reason': 'using_email_summary'
            }
        
        # Content is too short or unavailable
        return {
            'text': '',
            'is_full_content': False,
            'skip_llm': True,
            'reason': 'insufficient_content'
        }
    
    async def _store_article_with_embeddings(self, article: Dict[str, Any]):
        """Store article in database with embeddings with proper concurrency control"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Use database connection with proper isolation
                stored_count = await self._store_single_article_safe(article)
                
                if stored_count > 0:
                    self.logger.info(f"Successfully stored article: {article.get('title', 'Unknown')[:50]}...")
                    return
                else:
                    self.logger.warning(f"Article already exists or failed to store: {article.get('title', 'Unknown')[:50]}...")
                    return  # Don't retry if article already exists
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Storage attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"Failed to store article after {max_retries} attempts: {e}")
                    raise
    
    async def _store_single_article_safe(self, article: Dict[str, Any]) -> int:
        """Store single article with proper database concurrency handling"""
        import psycopg2
        from psycopg2.extras import RealDictCursor
        from backend.config import config
        from sentence_transformers import SentenceTransformer
        from datetime import datetime
        
        # Get or create embedding model instance (reuse across workers)
        if not hasattr(self, '_embedding_model'):
            self._embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        try:
            # Create new connection for this transaction
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME,
                cursor_factory=RealDictCursor
            )
            
            with conn:
                with conn.cursor() as cursor:
                    # Check if article already exists (atomic operation)
                    cursor.execute(
                        f"SELECT id FROM {config.VECTOR_TABLE_NAME} WHERE hash = %s",
                        (article['hash'],)
                    )
                    
                    if cursor.fetchone():
                        self.logger.debug(f"Article already exists: {article['title'][:50]}...")
                        return 0  # Already exists
                    
                    # Generate embedding from content for search
                    full_content = article.get('full_content', '')
                    if not full_content or 'Unable to fetch' in full_content:
                        embedding_text = f"{article.get('title', '')}\n\n{article.get('summary', '')}"
                    else:
                        embedding_text = full_content
                    
                    embedding = self._embedding_model.encode(embedding_text).tolist()
                    
                    # Store both original and LLM summary
                    original_summary = article.get('summary', '')
                    llm_summary = article.get('llm_summary', '')
                    
                    # Insert article with all data including LLM summary
                    cursor.execute(f"""
                        INSERT INTO {config.VECTOR_TABLE_NAME} 
                        (title, link, summary, llm_summary, content, author, hash, processed_at, embedding, digest_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (hash) DO NOTHING
                    """, (
                        article['title'][:500],
                        article['link'][:1000],
                        original_summary[:2000],     # Original summary
                        llm_summary[:2000],          # LLM-generated summary  
                        full_content,                # Full content for embedding
                        article.get('author', '')[:200],
                        article['hash'],
                        datetime.now(),
                        embedding,
                        article.get('digest_date')
                    ))
                    
                    # Check if row was actually inserted
                    if cursor.rowcount > 0:
                        return 1
                    else:
                        return 0  # Conflict occurred, article already exists
                        
        except psycopg2.IntegrityError as e:
            # Handle duplicate key violations gracefully
            self.logger.debug(f"Article already exists (integrity error): {article['title'][:50]}...")
            return 0
        except Exception as e:
            self.logger.error(f"Database error storing article: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current processing queue status"""
        return {
            "queue_size": self.processing_queue.qsize(),
            "active_workers": len([w for w in self.workers if not w.done()]),
            "total_workers": len(self.workers),
            "running": self.running
        }
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing tracking parameters and fragments."""
        try:
            from urllib.parse import urlparse, urlunparse, parse_qs
            
            parsed = urlparse(url)
            
            # Remove query parameters (tracking data)
            clean_query = ''
            
            # Remove fragment (anchors)
            clean_fragment = ''
            
            # Reconstruct clean URL
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                clean_query,
                clean_fragment
            ))
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"URL normalization failed for {url}: {e}")
            return url
    
    def _is_substantial_content(self, article: Dict[str, Any]) -> bool:
        """Check if article represents substantial content worth indexing."""
        title = article.get('title', '').lower()
        url = article.get('link', '')
        summary = article.get('summary', '')
        
        # Check for publication pages vs individual articles
        publication_indicators = [
            'data science collective',
            'artificial intelligence in plain english',
            'tech leadership unlocked',
            'control your recommendations',
            'switch to the weekly digest',
            'the medium newsletter',
            'social stories by product coalition'
        ]
        
        if any(indicator in title for indicator in publication_indicators):
            return False
        
        # Check for individual article indicators
        article_indicators = [
            'min read',  # "5 min read" indicates an article
            '…',         # Truncated titles suggest full articles
        ]
        
        has_article_indicators = any(indicator in title for indicator in article_indicators)
        
        # Check URL structure for individual articles
        individual_article_url = (
            '/@' in url and  # Author-based URL
            not url.endswith('?source=email') and  # Not just publication link
            len(url.split('/')) > 4  # Has path segments beyond domain
        )
        
        # Check content length
        has_substantial_summary = len(summary) > 50 and not summary.startswith('Summary for:')
        
        # Article is substantial if it has:
        # 1. Article indicators (reading time) OR
        # 2. Individual article URL structure AND substantial summary
        is_substantial = (
            has_article_indicators or 
            (individual_article_url and has_substantial_summary)
        )
        
        if not is_substantial:
            self.logger.debug(f"Non-substantial content: {title[:50]}... (URL: {url[:100]}...)")
        
        return is_substantial
    
    async def stop(self):
        """Stop the content agent and workers"""
        self.logger.info("Stopping Content Agent...")
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.logger.info("Content Agent stopped")

# Usage example:
content_agent = ContentAgent("content_agent_001", max_workers=3)
