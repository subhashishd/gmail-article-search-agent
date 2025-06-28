"""
Direct Fetch Service - Bypasses multi-agent complexity for reliable operation.

This service provides:
1. Immediate API response with background processing
2. Parallel content fetching with rate limiting
3. Real-time progress tracking
4. Proper concurrency control
"""

import asyncio
import hashlib
import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from concurrent.futures import ThreadPoolExecutor
import functools

from backend.config import config
from backend.services.memory_service import memory_service
from backend.services.article_processing_service import ArticleContentFetcher
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("DirectFetchService")

class DirectFetchService:
    """Direct fetch service that works reliably without multi-agent complexity."""
    
    def __init__(self):
        self.active_operations = {}
        self.lock = asyncio.Lock()
        self.content_fetcher = ArticleContentFetcher()
        self.embedding_model = None
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gmail-fetch")
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize embedding model for article processing."""
        try:
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.embedding_model = None
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            if self.embedding_model is None:
                logger.warning("Embedding model not loaded, returning zero embedding")
                return [0.0] * 384  # Default dimension
            
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 384
    
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
    
    async def start_fetch_operation(self, max_emails: int = 5) -> Dict[str, Any]:
        """Start a fetch operation - returns immediately with operation ID."""
        async with self.lock:
            # Check if operation is already running
            active_ops = [op for op in self.active_operations.values() 
                         if op["status"] in ["starting", "running"]]
            
            if active_ops:
                existing_op = active_ops[0]
                return {
                    "success": False,
                    "message": "Fetch operation already in progress",
                    "operation_id": existing_op["operation_id"],
                    "status": existing_op["status"]
                }
            
            # Create new operation
            operation_id = f"fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            operation = {
                "operation_id": operation_id,
                "status": "starting",
                "started_at": datetime.now(),
                "progress": {
                    "emails_processed": 0,
                    "total_emails": 0,
                    "articles_found": 0,
                    "content_fetched": 0,
                    "articles_stored": 0,
                    "errors": 0
                },
                "current_step": "initializing",
                "message": "Starting fetch operation...",
                "max_emails": max_emails
            }
            
            self.active_operations[operation_id] = operation
            
            # Start background task
            asyncio.create_task(self._execute_fetch_operation(operation_id))
            
            return {
                "success": True,
                "message": "Fetch operation started",
                "operation_id": operation_id,
                "status": "starting",
                "estimated_duration": "1-3 minutes"
            }
    
    async def _execute_fetch_operation(self, operation_id: str):
        """
        NEW BULK ARCHITECTURE: Execute fetch operation with bulk email processing.
        
        This method:
        1. Bulk fetches all Gmail emails quickly (single API call)
        2. Extracts article metadata from all emails in-memory (fast)
        3. Processes content fetching and storage in parallel
        """
        operation = self.active_operations[operation_id]
        
        try:
            operation["status"] = "running"
            operation["current_step"] = "bulk_fetching_emails"
            operation["message"] = "Bulk fetching emails and extracting article metadata..."
            
            # Get last update time
            last_update = memory_service.get_last_update_time()
            logger.info(f"Bulk fetching emails since: {last_update}")
            
            # Step 1: Bulk fetch all emails from Gmail (single fast operation)
            try:
                emails = await asyncio.wait_for(
                    self._fetch_emails_async(last_update),
                    timeout=300.0  # 5 minute timeout for Gmail API (large email sets)
                )
            except asyncio.TimeoutError:
                operation["status"] = "failed"
                operation["error"] = "Gmail bulk fetch timeout after 5 minutes"
                operation["message"] = "Gmail bulk fetch timed out"
                return
            except Exception as e:
                operation["status"] = "failed"
                operation["error"] = str(e)
                operation["message"] = f"Gmail bulk fetch error: {str(e)}"
                return
            
            if not emails:
                operation["status"] = "completed"
                operation["message"] = "No new emails found"
                operation["completed_at"] = datetime.now()
                return
            
            # Limit emails if specified
            if operation["max_emails"]:
                emails = emails[:operation["max_emails"]]
            
            # Sort emails chronologically
            emails = sorted(emails, key=lambda m: m['date'], reverse=False)
            
            operation["progress"]["total_emails"] = len(emails)
            operation["current_step"] = "bulk_extracting_articles"
            operation["message"] = f"Bulk extracting articles from {len(emails)} emails..."
            
            logger.info(f"Bulk processing {len(emails)} emails for article extraction...")
            
            # Step 2: Bulk extract articles from all emails (fast in-memory operations)
            articles_dataset = []
            latest_email_date = None
            
            for email_idx, email in enumerate(emails, 1):
                try:
                    email_date = email['date']
                    
                    # Track latest email date for timestamp update
                    if latest_email_date is None or email_date > latest_email_date:
                        latest_email_date = email_date
                    
                    # Extract articles from email (fast, in-memory operation)
                    articles = await asyncio.wait_for(
                        self._extract_articles_async(email),
                        timeout=30.0  # Reduced timeout for fast extraction
                    )
                    
                    # Add metadata to articles
                    for article in articles:
                        article['digest_date'] = email_date.date()
                        article['email_date'] = email_date
                        article['email_index'] = email_idx
                        # Note: full_content will be fetched later in parallel
                        article['full_content'] = None  # Placeholder for parallel fetching
                    
                    articles_dataset.extend(articles)
                    operation["progress"]["articles_found"] += len(articles)
                    
                    # Update progress for status tracking
                    operation["progress"]["emails_processed"] = email_idx
                    
                    # Log progress every 10 emails
                    if email_idx % 10 == 0:
                        logger.info(f"Processed {email_idx}/{len(emails)} emails, found {operation['progress']['articles_found']} articles")
                        operation["message"] = f"Bulk extracting articles: {email_idx}/{len(emails)} emails processed, {operation['progress']['articles_found']} articles found"
                    
                    # Small delay to yield control and allow status checks
                    await asyncio.sleep(0.05)
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout extracting articles from email {email_idx}")
                    operation["progress"]["errors"] += 1
                except Exception as e:
                    logger.error(f"Error extracting articles from email {email_idx}: {e}")
                    operation["progress"]["errors"] += 1
            
            # Update progress after bulk extraction
            operation["progress"]["emails_processed"] = len(emails)
            
            if not articles_dataset:
                operation["status"] = "completed"
                operation["message"] = "No articles found in emails"
                operation["completed_at"] = datetime.now()
                # Still update timestamp for processed emails
                if latest_email_date:
                    memory_service.save_last_update_time(latest_email_date)
                return
            
            logger.info(f"Bulk extraction complete: {len(emails)} emails processed, {len(articles_dataset)} articles extracted")
            
            # Step 3: Parallel content fetching and database storage from in-memory dataset
            operation["current_step"] = "parallel_processing"
            operation["message"] = f"Processing {len(articles_dataset)} articles in parallel..."
            
            stored_count = await self._parallel_content_processing(articles_dataset, operation)
            
            # Step 4: Update timestamp after all processing is complete
            if latest_email_date:
                memory_service.save_last_update_time(latest_email_date)
                logger.info(f"Updated last processed time to: {latest_email_date}")
            
            operation["progress"]["articles_stored"] = stored_count
            operation["status"] = "completed"
            operation["completed_at"] = datetime.now()
            operation["message"] = f"Successfully processed {len(emails)} emails and stored {stored_count} articles"
            
        except Exception as e:
            logger.error(f"Error in fetch operation {operation_id}: {e}")
            operation["status"] = "failed"
            operation["error"] = str(e)
            operation["message"] = f"Fetch operation failed: {str(e)}"
    
    async def _parallel_content_processing(self, articles: List[Dict], operation: Dict) -> int:
        """Process article content fetching in parallel with rate limiting."""
        max_concurrent = 5  # Limit concurrent requests to Medium
        semaphore = asyncio.Semaphore(max_concurrent)
        stored_count = 0
        
        async def process_single_article(article):
            async with semaphore:
                try:
                    # Fetch content using thread executor with timeout
                    full_content = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            self.content_fetcher.fetch_article_content,
                            article['link']
                        ),
                        timeout=20.0  # 20 second timeout per article
                    )
                    article['full_content'] = full_content
                    operation["progress"]["content_fetched"] += 1
                    
                    return article
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout fetching content for {article.get('link', 'unknown')}")
                    article['full_content'] = "Content fetch timeout"
                    operation["progress"]["errors"] += 1
                    return article
                except Exception as e:
                    logger.error(f"Error fetching content for {article.get('link', 'unknown')}: {e}")
                    article['full_content'] = f"Unable to fetch content: {str(e)}"
                    operation["progress"]["errors"] += 1
                    return article
        
        # Process all articles in parallel
        operation["message"] = f"Fetching content for {len(articles)} articles..."
        processed_articles = await asyncio.gather(
            *[process_single_article(article) for article in articles],
            return_exceptions=True
        )
        
        # Filter out exceptions
        valid_articles = [a for a in processed_articles if not isinstance(a, Exception)]
        
        # Store in database
        if valid_articles:
            operation["current_step"] = "storing_articles"
            operation["message"] = f"Storing {len(valid_articles)} articles in database..."
            
            try:
                stored_count = await self._store_articles_batch(valid_articles)
                logger.info(f"Stored {stored_count} articles in database")
            except Exception as e:
                logger.error(f"Error storing articles: {e}")
                operation["progress"]["errors"] += 1
        
        return stored_count
    
    async def _store_articles_batch(self, articles: List[Dict]) -> int:
        """Store articles in database with embeddings using individual transactions."""
        logger.info(f"DEBUG: _store_articles_batch called with {len(articles)} articles")
        stored_count = 0
        
        for article in articles:
            conn = None
            try:
                # Use individual connection/transaction for each article to prevent rollback issues
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Check if article already exists
                cursor.execute(
                    f"SELECT id FROM {config.VECTOR_TABLE_NAME} WHERE hash = %s",
                    (article['hash'],)
                )
                if cursor.fetchone():
                    logger.info(f"Article already exists: {article['title'][:50]}...")
                    cursor.close()
                    conn.close()
                    continue
                
                # Get content for embedding
                full_content = article.get('full_content', '')
                if not full_content or 'Unable to fetch' in full_content:
                    # Use title and summary as fallback
                    full_content = f"{article.get('title', '')}\n\n{article.get('summary', '')}"
                
                # Generate embedding
                embedding = self._generate_embedding(full_content)
                
                # Insert article (fixed to match actual schema)
                sql_statement = f"""
                    INSERT INTO {config.VECTOR_TABLE_NAME} 
                    (title, link, summary, content, author, hash, processed_at, embedding, digest_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                logger.info(f"DEBUG: Executing SQL: {sql_statement}")
                cursor.execute(sql_statement, (
                    article['title'][:500],
                    article['link'][:1000],
                    full_content[:2000],  # Summary from first 2000 chars
                    full_content,         # Full content
                    article.get('author', '')[:200],
                    article['hash'],
                    datetime.now(),
                    embedding,
                    article.get('digest_date')
                ))
                
                conn.commit()
                cursor.close()
                conn.close()
                
                stored_count += 1
                logger.info(f"Stored article: {article['title'][:50]}...")
                
            except Exception as e:
                logger.error(f"Error storing article {article.get('title', 'Unknown')}: {e}")
                if conn:
                    try:
                        conn.rollback()
                        conn.close()
                    except:
                        pass
                continue
        
        logger.info(f"Batch storage complete: {stored_count}/{len(articles)} articles stored")
        return stored_count
    
    def get_operation_status(self, operation_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of fetch operation(s)."""
        try:
            if operation_id:
                operation = self.active_operations.get(operation_id)
                if not operation:
                    return {"error": f"Operation {operation_id} not found"}
                
                # Add duration calculation safely
                try:
                    if "started_at" in operation:
                        duration = (datetime.now() - operation["started_at"]).total_seconds()
                        operation = operation.copy()
                        operation["duration_seconds"] = duration
                except Exception as e:
                    logger.warning(f"Error calculating duration: {e}")
                    operation = operation.copy()
                    operation["duration_seconds"] = 0
                
                return operation
            
            # Return latest operation status
            if not self.active_operations:
                return {
                    "status": "idle",
                    "message": "No fetch operations found"
                }
            
            # Get most recent operation safely
            try:
                latest_op = max(self.active_operations.values(), key=lambda x: x.get("started_at", datetime.min))
            except Exception as e:
                logger.error(f"Error getting latest operation: {e}")
                return {
                    "status": "error",
                    "message": "Error retrieving operation status"
                }
            
            # Add duration calculation safely
            try:
                if "started_at" in latest_op:
                    duration = (datetime.now() - latest_op["started_at"]).total_seconds()
                    latest_op = latest_op.copy()
                    latest_op["duration_seconds"] = duration
            except Exception as e:
                logger.warning(f"Error calculating duration: {e}")
                latest_op = latest_op.copy()
                latest_op["duration_seconds"] = 0
            
            return latest_op
            
        except Exception as e:
            logger.error(f"Error in get_operation_status: {e}")
            return {
                "status": "error",
                "message": f"Status retrieval error: {str(e)}"
            }
    
    async def _fetch_emails_async(self, last_update):
        """Fetch emails from Gmail using thread executor to prevent blocking."""
        def _gmail_fetch():
            # Import Gmail service here to avoid initialization issues
            from backend.services.gmail_service_oauth import get_gmail_service
            gmail_service = get_gmail_service()
            return gmail_service.mcp_service.search_medium_emails(last_update=last_update)
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _gmail_fetch)
    
    async def _extract_articles_async(self, email):
        """Extract articles from email using thread executor to prevent blocking."""
        def _extract_articles():
            # Import Gmail service here to avoid initialization issues
            from backend.services.gmail_service_oauth import get_gmail_service
            gmail_service = get_gmail_service()
            return gmail_service.mcp_service.get_articles_from_email(email)
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _extract_articles)
    
    def cleanup_old_operations(self, max_age_hours: int = 24):
        """Clean up old operation records."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        to_remove = []
        for op_id, operation in self.active_operations.items():
            if operation["started_at"].timestamp() < cutoff_time:
                to_remove.append(op_id)
        
        for op_id in to_remove:
            del self.active_operations[op_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old operations")

# Global service instance
direct_fetch_service = DirectFetchService()
