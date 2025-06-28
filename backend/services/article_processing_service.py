
"""Service for processing articles from Gmail and storing them in the database."""

import requests
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import List, Dict
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# Lazy imports to prevent automatic initialization
from backend.services.memory_service import memory_service
from backend.config import config

class ArticleContentFetcher:
    """Fetches full content from Medium articles."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def fetch_article_content(self, url: str) -> str:
        """Fetch full article content from Medium URL."""
        try:
            print(f"[CONTENT] Fetching full content from: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try different selectors for Medium content
            content_selectors = [
                'article',  # Main article tag
                '[data-testid="storyContent"]',  # New Medium structure
                '.postArticle-content',  # Old Medium structure
                '.section-content',  # Alternative structure
                'section'  # Fallback
            ]
            
            article_content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # Get text from all paragraphs within the selected content
                    paragraphs = elements[0].find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if paragraphs:
                        article_content = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                        break
            
            # Fallback: get all paragraphs from the page
            if not article_content:
                paragraphs = soup.find_all('p')
                article_content = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            # Clean up and limit content
            if article_content:
                # Remove extra whitespace
                import re
                article_content = re.sub(r'\s+', ' ', article_content).strip()
                # Limit to reasonable size for embeddings (about 8000 words)
                if len(article_content) > 50000:
                    article_content = article_content[:50000] + "... [content truncated]"
                print(f"[CONTENT] Fetched {len(article_content)} characters of content")
                return article_content
            else:
                print(f"[CONTENT] No content found, using fallback")
                return f"Content not available for: {url}"
                
        except Exception as e:
            print(f"[CONTENT] Error fetching content from {url}: {e}")
            return f"Unable to fetch full content from: {url}"

class ArticleProcessingService:
    def __init__(self):
        self.content_fetcher = ArticleContentFetcher()
        self.embedding_model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model for embeddings."""
        try:
            print(f"📦 Loading embedding model: {config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.embedding_model = None
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            if self.embedding_model is None:
                print("⚠️ Model not loaded, returning zero embedding")
                return [0.0] * self.embedding_dim
            
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return [0.0] * self.embedding_dim
    
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
    
    def _store_article_with_full_content(self, article: Dict, full_content: str) -> bool:
        """Store article with full content and embeddings directly in database."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Check if article already exists
            cursor.execute(
                f"SELECT id FROM {config.VECTOR_TABLE_NAME} WHERE hash = %s",
                (article['hash'],)
            )
            if cursor.fetchone():
                print(f"⚠️ Article already exists: {article['title'][:50]}...")
                cursor.close()
                conn.close()
                return True
            
            # Generate embedding from full content
            print(f"🧠 Generating embedding for full content ({len(full_content)} chars)...")
            embedding = self._generate_embedding(full_content)
            
            # Insert article with full content (fixed to match actual schema)
            cursor.execute(f"""
                INSERT INTO {config.VECTOR_TABLE_NAME} 
                (title, link, summary, content, author, hash, processed_at, embedding, digest_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                article['title'][:500],  # Limit title length
                article['link'][:1000],  # Limit link length  
                full_content[:2000],     # Summary from first 2000 chars of content
                full_content,            # Store FULL content
                article.get('author', '')[:200],  # Author info
                article['hash'],
                datetime.now(),
                embedding,
                article.get('digest_date')
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"💾 Stored article with full content: {article['title'][:50]}...")
            return True
            
        except Exception as e:
            print(f"❌ Error storing article: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    async def process_articles(self) -> Dict:
        """Fetches, processes, and stores articles from Gmail chronologically with full content."""
        # DISABLED: This method should only be called manually via API
        print("⚠️ process_articles() called - this should only happen via manual trigger")
        
        # Lazy import to prevent automatic initialization
        from backend.services.gmail_service_oauth import get_gmail_service
        gmail_service = get_gmail_service()
        
        last_update = memory_service.get_last_update_time()
        print(f"🕐 Fetching articles since: {last_update}")

        try:
            # Fetch new emails
            messages = gmail_service.mcp_service.search_medium_emails(last_update=last_update)
            if not messages:
                return {"message": "No new articles found."}

            print(f"📧 Found {len(messages)} new emails to process.")
            
            # Sort emails chronologically (oldest first) - THIS IS CRITICAL
            sorted_messages = sorted(messages, key=lambda m: m['date'], reverse=False)
            
            print(f"📅 Processing emails chronologically from {sorted_messages[0]['date']} to {sorted_messages[-1]['date']}")

            # Count total articles first
            all_articles = []
            for message in sorted_messages:
                articles = gmail_service.mcp_service.get_articles_from_email(message)
                all_articles.extend(articles)

            total_articles = len(all_articles)
            print(f"📚 Found {total_articles} total articles to process with full content fetching.")

            if total_articles == 0:
                return {"message": "No articles found in emails."}

            processed_count = 0
            failed_count = 0
            
            # Process emails chronologically
            for email_idx, message in enumerate(sorted_messages, 1):
                email_date = message['date']
                print(f"\n📧 Processing email {email_idx}/{len(sorted_messages)} from {email_date}")
                
                articles = gmail_service.mcp_service.get_articles_from_email(message)
                
                if articles:
                    print(f"📄 Processing {len(articles)} articles from this email...")
                    
                    # Process each article with full content fetching
                    for article in articles:
                        processed_count += 1
                        try:
                            print(f"\n🔍 Processing article {processed_count}/{total_articles}: {article['title'][:60]}...")
                            
                            # Fetch full content from Medium
                            full_content = self.content_fetcher.fetch_article_content(article['link'])
                            
                            # Update article with digest date
                            article['digest_date'] = email_date.date()
                            
                            # Store article with full content directly in database
                            success = self._store_article_with_full_content(article, full_content)
                            if success:
                                print(f"✅ Successfully processed with full content: {article['title'][:60]}...")
                            else:
                                failed_count += 1
                                print(f"❌ Failed to store: {article['title'][:60]}...")
                                
                        except Exception as article_error:
                            failed_count += 1
                            print(f"❌ Error processing article '{article.get('title', 'Unknown')}': {article_error}")
                            continue
                
                # Update last update time after processing each email (for restart recovery)
                memory_service.save_last_update_time(email_date)
                print(f"💾 Updated last update time to: {email_date}")

            success_count = processed_count - failed_count
            result_message = f"📊 Processing complete! Successfully processed {success_count}/{total_articles} articles from {len(messages)} emails."
            
            if failed_count > 0:
                result_message += f" Failed: {failed_count} articles."
                
            print(f"\n🎉 {result_message}")
            return {"message": result_message}

        except Exception as e:
            error_msg = f"Error processing articles: {e}"
            print(f"💥 {error_msg}")
            return {"error": error_msg}

async def store_articles_batch(articles: List[Dict]) -> int:
    """Store multiple articles in batch with embeddings - optimized for parallel processing."""
    if not articles:
        return 0
    
    service = ArticleProcessingService()
    stored_count = 0
    
    try:
        conn = service._get_db_connection()
        cursor = conn.cursor()
        
        for article in articles:
            try:
                # Check if article already exists
                cursor.execute(
                    f"SELECT id FROM {config.VECTOR_TABLE_NAME} WHERE hash = %s",
                    (article['hash'],)
                )
                if cursor.fetchone():
                    print(f"⚠️ Article already exists: {article['title'][:50]}...")
                    continue
                
                # Get content (should already be fetched in parallel)
                full_content = article.get('full_content', '')
                if not full_content or 'Unable to fetch' in full_content or 'Processing error' in full_content:
                    # Use title and summary as fallback
                    full_content = f"{article.get('title', '')}\n\n{article.get('summary', '')}"
                
                # Generate embedding from content
                print(f"🧠 Generating embedding for: {article['title'][:50]}...")
                embedding = service._generate_embedding(full_content)
                
                # Insert article (fixed to match actual schema)
                cursor.execute(f"""
                    INSERT INTO {config.VECTOR_TABLE_NAME} 
                    (title, link, summary, content, author, hash, processed_at, embedding, digest_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    article['title'][:500],  # Limit title length
                    article['link'][:1000],  # Limit link length  
                    full_content[:2000],     # Summary from first 2000 chars of content
                    full_content,            # Store FULL content
                    article.get('author', '')[:200],  # Author info
                    article['hash'],
                    datetime.now(),
                    embedding,
                    article.get('digest_date')
                ))
                
                stored_count += 1
                print(f"💾 Stored article: {article['title'][:50]}...")
                
            except Exception as e:
                print(f"❌ Error storing individual article {article.get('title', 'Unknown')}: {e}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Batch storage complete: {stored_count}/{len(articles)} articles stored")
        return stored_count
        
    except Exception as e:
        print(f"❌ Error in batch storage: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return stored_count

article_processing_service = ArticleProcessingService()
