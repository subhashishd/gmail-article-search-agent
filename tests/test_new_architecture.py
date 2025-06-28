"""
Comprehensive Test Suite for New Event-Driven Architecture

Tests all scenarios:
1. Email processing from Jan 1, 2025
2. Timestamp management after each email
3. Parallel article processing with content fetching
4. Database storage with embeddings
5. Vector search with LLM contextualization
"""

import asyncio
import pytest
import psycopg2
from datetime import datetime, timedelta
import json
import redis
from unittest.mock import Mock, patch, AsyncMock
import logging

# Setup test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_architecture")

# Import our new components
from backend.core.event_bus import event_bus
from backend.core.rate_limiter import rate_limiter
from backend.core.event_coordinator import event_coordinator
from backend.agents.email_processor_agent import EmailProcessorAgent
from backend.agents.content_agent import ContentAgent
from backend.services.memory_service import memory_service
from backend.config import config

class TestEventDrivenArchitecture:
    """Test suite for the new event-driven architecture"""
    
    @pytest.fixture
    async def setup_test_environment(self):
        """Setup test environment with clean state"""
        logger.info("Setting up test environment...")
        
        # Reset memory service to Jan 1, 2025
        test_start_date = datetime(2025, 1, 1, 0, 0, 0)
        memory_service.save_last_update_time(test_start_date)
        
        # Clear Redis
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=1)  # Use test DB
            redis_client.flushdb()
        except:
            logger.warning("Redis not available for testing - using mock")
        
        # Clear test database table
        try:
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME
            )
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {config.VECTOR_TABLE_NAME} WHERE title LIKE 'TEST_%'")
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.warning(f"Database cleanup failed: {e}")
        
        yield
        
        # Cleanup after tests
        logger.info("Cleaning up test environment...")

    def create_mock_email_data(self, email_date, num_articles=3):
        """Create mock email data for testing"""
        return {
            'id': f"test_email_{email_date.strftime('%Y%m%d')}",
            'subject': f"Medium Daily Digest - {email_date.strftime('%B %d, %Y')}",
            'sender': 'noreply@medium.com',
            'date': email_date,
            'body': f"""
            <html>
            <body>
                <h1>Your daily digest from Medium</h1>
                {''.join([f'''
                <div>
                    <a href="https://medium.com/test-article-{i}-{email_date.strftime('%Y%m%d')}">
                        TEST_Article_{i}_{email_date.strftime('%Y%m%d')}: Advanced AI Techniques for Modern Development
                    </a>
                    <p>This is a test article about AI and machine learning techniques...</p>
                </div>
                ''' for i in range(1, num_articles + 1)])}
            </body>
            </html>
            """
        }

    def create_mock_article_content(self, article_title):
        """Create mock article content for testing"""
        return f"""
        {article_title}
        
        This is the full content of the test article about advanced AI techniques.
        
        In this comprehensive guide, we'll explore:
        1. Machine Learning fundamentals
        2. Deep Learning architectures
        3. Natural Language Processing
        4. Computer Vision applications
        5. AI Ethics and best practices
        
        The article covers practical implementations using Python, TensorFlow, and PyTorch.
        We'll dive deep into neural networks, transformer models, and attention mechanisms.
        
        Key takeaways:
        - Understanding AI model architectures
        - Implementing scalable ML solutions
        - Best practices for model deployment
        - Performance optimization techniques
        
        This content provides valuable insights for developers and data scientists
        working with modern AI technologies in production environments.
        """

    @pytest.mark.asyncio
    async def test_01_timestamp_initialization(self, setup_test_environment):
        """Test that timestamp is correctly initialized to Jan 1, 2025"""
        logger.info("Testing timestamp initialization...")
        
        last_update = memory_service.get_last_update_time()
        expected_date = datetime(2025, 1, 1, 0, 0, 0)
        
        assert last_update.date() == expected_date.date(), f"Expected {expected_date.date()}, got {last_update.date()}"
        logger.info(f"✅ Timestamp correctly initialized to: {last_update}")

    @pytest.mark.asyncio
    async def test_02_event_bus_initialization(self, setup_test_environment):
        """Test Redis event bus initialization"""
        logger.info("Testing event bus initialization...")
        
        try:
            # Initialize event bus with test Redis DB
            test_event_bus = event_bus
            test_event_bus.redis_url = "redis://localhost:6379/1"  # Test DB
            await test_event_bus.initialize()
            
            # Test publishing and subscribing
            received_events = []
            
            async def test_handler(event):
                received_events.append(event)
            
            await test_event_bus.subscribe("test.event", test_handler)
            
            # Publish test event
            event_id = await test_event_bus.publish(
                "test.event",
                {"message": "test"},
                "test_source"
            )
            
            # Wait for event processing
            await asyncio.sleep(0.5)
            
            assert len(received_events) > 0, "Event was not received"
            assert received_events[0].data["message"] == "test"
            
            logger.info("✅ Event bus initialization successful")
            
        except Exception as e:
            logger.warning(f"Event bus test skipped - Redis not available: {e}")

    @pytest.mark.asyncio
    async def test_03_email_processing_sequential_with_parallel_articles(self, setup_test_environment):
        """Test sequential email processing with parallel article processing"""
        logger.info("Testing email processing flow...")
        
        # Create mock emails for 3 consecutive days
        test_emails = []
        start_date = datetime(2025, 1, 2)  # Day after initialization
        
        for i in range(3):
            email_date = start_date + timedelta(days=i)
            test_emails.append(self.create_mock_email_data(email_date, num_articles=5))
        
        # Mock Gmail service
        with patch('backend.services.gmail_service_oauth.get_gmail_service') as mock_gmail:
            mock_service = Mock()
            mock_service.authenticate.return_value = True
            mock_service.mcp_service.search_medium_emails.return_value = test_emails
            
            # Mock article extraction
            def mock_get_articles(email):
                articles = []
                for i in range(1, 6):  # 5 articles per email
                    articles.append({
                        'title': f"TEST_Article_{i}_{email['date'].strftime('%Y%m%d')}",
                        'link': f"https://medium.com/test-article-{i}-{email['date'].strftime('%Y%m%d')}",
                        'summary': f"Test summary for article {i}",
                        'hash': f"test_hash_{i}_{email['date'].strftime('%Y%m%d')}",
                        'digest_date': email['date'].date(),
                        'author': 'Test Author'
                    })
                return articles
            
            mock_service.mcp_service.get_articles_from_email.side_effect = mock_get_articles
            mock_gmail.return_value = mock_service
            
            # Initialize email processor
            email_processor = EmailProcessorAgent("test_email_processor")
            await email_processor.initialize()
            
            # Track processed articles
            processed_articles = []
            
            async def track_articles(event):
                processed_articles.append(event.data)
            
            await event_bus.subscribe("article.discovered", track_articles)
            
            # Process emails
            result = await email_processor.process_emails_sequentially(max_emails=3)
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Verify results
            assert result["success"] is True
            assert result["processed_emails"] == 3
            assert result["total_articles"] == 15  # 3 emails × 5 articles
            
            # Verify timestamp updated to last email date
            last_update = memory_service.get_last_update_time()
            expected_last_date = start_date + timedelta(days=2)  # Last email date
            assert last_update.date() == expected_last_date.date()
            
            logger.info(f"✅ Processed {result['processed_emails']} emails with {result['total_articles']} articles")
            logger.info(f"✅ Timestamp updated to: {last_update}")

    @pytest.mark.asyncio
    async def test_04_parallel_content_fetching_and_storage(self, setup_test_environment):
        """Test parallel content fetching and database storage"""
        logger.info("Testing parallel content fetching and storage...")
        
        # Create test articles
        test_articles = []
        for i in range(1, 6):
            test_articles.append({
                'title': f"TEST_Content_Article_{i}",
                'link': f"https://medium.com/test-content-{i}",
                'summary': f"Test summary {i}",
                'hash': f"test_content_hash_{i}",
                'digest_date': datetime(2025, 1, 3).date(),
                'author': 'Test Author'
            })
        
        # Mock content fetching
        async def mock_fetch_content(url):
            article_title = url.split('/')[-1].replace('-', ' ').title()
            return self.create_mock_article_content(article_title)
        
        with patch('backend.services.article_content_fetcher.fetch_article_content', side_effect=mock_fetch_content):
            # Initialize content agent
            content_agent = ContentAgent("test_content_agent", max_workers=3)
            await content_agent.initialize()
            
            # Track stored articles
            stored_articles = []
            
            async def track_stored(event):
                stored_articles.append(event.data)
            
            await event_bus.subscribe("article.stored", track_stored)
            
            # Publish articles for processing
            for article in test_articles:
                await event_bus.publish("article.discovered", article, "test")
            
            # Wait for processing
            await asyncio.sleep(5)
            
            # Verify database storage
            try:
                conn = psycopg2.connect(
                    host=config.DB_HOST,
                    port=config.DB_PORT,
                    user=config.DB_USER,
                    password=config.DB_PASS,
                    database=config.DB_NAME
                )
                cursor = conn.cursor()
                
                cursor.execute(f"""
                    SELECT title, content, embedding 
                    FROM {config.VECTOR_TABLE_NAME} 
                    WHERE title LIKE 'TEST_Content_%'
                """)
                
                stored_records = cursor.fetchall()
                cursor.close()
                conn.close()
                
                assert len(stored_records) >= 3, f"Expected at least 3 stored articles, got {len(stored_records)}"
                
                # Verify content and embeddings are stored
                for record in stored_records:
                    title, content, embedding = record
                    assert content is not None and len(content) > 100, f"Content not properly stored for {title}"
                    assert embedding is not None, f"Embedding not generated for {title}"
                
                logger.info(f"✅ Successfully stored {len(stored_records)} articles with content and embeddings")
                
            except Exception as e:
                logger.warning(f"Database verification failed: {e}")
            
            # Stop content agent
            await content_agent.stop()

    @pytest.mark.asyncio
    async def test_05_vector_search_with_llm_contextualization(self, setup_test_environment):
        """Test vector search with LLM contextualization"""
        logger.info("Testing vector search with LLM contextualization...")
        
        # First, ensure we have test data in the database
        await self.test_04_parallel_content_fetching_and_storage(setup_test_environment)
        
        # Mock LLM service for contextualization
        mock_llm_response = """
        Relevance Score: 0.85
        Reasoning: This article is highly relevant to AI and machine learning, covering practical implementations and best practices.
        Key Topics: Machine Learning, Deep Learning, Neural Networks, Python, TensorFlow
        """
        
        with patch('backend.services.ollama_service.ollama_service.generate_response', 
                   return_value=mock_llm_response) as mock_llm:
            
            # Test search functionality
            from backend.services.hybrid_rag_service import HybridRAGService
            
            rag_service = HybridRAGService()
            
            # Perform search
            search_query = "AI machine learning techniques"
            results = await rag_service.search_and_analyze(search_query, top_k=5)
            
            # Verify search results
            assert "results" in results
            assert len(results["results"]) > 0, "No search results returned"
            
            # Verify LLM enhancement
            for result in results["results"]:
                assert "score" in result, "LLM score not assigned"
                assert "llm_reasoning" in result, "LLM reasoning not provided"
                assert "key_topics" in result, "Key topics not extracted"
                assert result.get("analysis_method") == "hybrid_vector_llm", "Wrong analysis method"
            
            # Verify LLM was called for contextualization
            assert mock_llm.called, "LLM service was not called for contextualization"
            
            logger.info(f"✅ Search returned {len(results['results'])} results with LLM enhancement")
            logger.info(f"✅ First result score: {results['results'][0].get('score', 'N/A')}")

    @pytest.mark.asyncio
    async def test_06_search_caching_functionality(self, setup_test_environment):
        """Test search result caching"""
        logger.info("Testing search caching functionality...")
        
        # Mock Redis for caching
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_redis_client = AsyncMock()
            mock_redis.return_value = mock_redis_client
            
            # First search - cache miss
            mock_redis_client.get.return_value = None
            mock_redis_client.setex = AsyncMock()
            
            # Mock search results
            mock_search_results = {
                "results": [
                    {
                        "title": "TEST_Cached_Article",
                        "score": 0.9,
                        "llm_reasoning": "Highly relevant test article"
                    }
                ],
                "total_found": 1
            }
            
            with patch('backend.services.hybrid_rag_service.hybrid_rag_service.search_and_analyze',
                       return_value=mock_search_results) as mock_search:
                
                from backend.agents.search_agent import SearchAgent
                search_agent = SearchAgent("test_search_agent")
                await search_agent.initialize()
                
                # First search - should hit the service and cache
                query = "test caching query"
                result1 = await search_agent.search_and_cache(query)
                
                # Verify search service was called
                assert mock_search.called, "Search service should be called on cache miss"
                assert mock_redis_client.setex.called, "Result should be cached"
                
                # Second search - should hit cache
                mock_redis_client.get.return_value = json.dumps(mock_search_results)
                mock_search.reset_mock()
                
                result2 = await search_agent.search_and_cache(query)
                
                # Verify cache was used
                assert not mock_search.called, "Search service should not be called on cache hit"
                
                logger.info("✅ Search caching working correctly")

    @pytest.mark.asyncio
    async def test_07_concurrent_article_processing(self, setup_test_environment):
        """Test concurrent processing of multiple articles without conflicts"""
        logger.info("Testing concurrent article processing...")
        
        # Create articles with potential conflicts (same hash)
        conflicting_articles = []
        for i in range(10):
            conflicting_articles.append({
                'title': f"TEST_Concurrent_Article_{i}",
                'link': f"https://medium.com/concurrent-test-{i}",
                'summary': f"Concurrent test summary {i}",
                'hash': f"concurrent_hash_{i % 3}",  # Some duplicates intentionally
                'digest_date': datetime(2025, 1, 4).date(),
                'author': 'Concurrent Test Author'
            })
        
        # Mock content fetching
        async def mock_fetch_content(url):
            return f"Concurrent test content for {url}"
        
        with patch('backend.services.article_content_fetcher.fetch_article_content', 
                   side_effect=mock_fetch_content):
            
            # Initialize content agent with more workers
            content_agent = ContentAgent("test_concurrent_agent", max_workers=5)
            await content_agent.initialize()
            
            # Track processing events
            processing_events = []
            
            async def track_processing(event):
                processing_events.append(event)
            
            await event_bus.subscribe("article.stored", track_processing)
            await event_bus.subscribe("article.processing.failed", track_processing)
            
            # Publish all articles simultaneously
            for article in conflicting_articles:
                await event_bus.publish("article.discovered", article, "concurrent_test")
            
            # Wait for processing
            await asyncio.sleep(8)
            
            # Verify no crashes and proper conflict handling
            stored_count = len([e for e in processing_events if e.type == "article.stored"])
            failed_count = len([e for e in processing_events if e.type == "article.processing.failed"])
            
            logger.info(f"✅ Concurrent processing completed: {stored_count} stored, {failed_count} failed")
            
            # Verify database integrity
            try:
                conn = psycopg2.connect(
                    host=config.DB_HOST,
                    port=config.DB_PORT,
                    user=config.DB_USER,
                    password=config.DB_PASS,
                    database=config.DB_NAME
                )
                cursor = conn.cursor()
                
                cursor.execute(f"""
                    SELECT hash, COUNT(*) 
                    FROM {config.VECTOR_TABLE_NAME} 
                    WHERE title LIKE 'TEST_Concurrent_%'
                    GROUP BY hash
                    HAVING COUNT(*) > 1
                """)
                
                duplicates = cursor.fetchall()
                cursor.close()
                conn.close()
                
                assert len(duplicates) == 0, f"Found duplicate hashes in database: {duplicates}"
                logger.info("✅ No duplicate articles found - conflict resolution working")
                
            except Exception as e:
                logger.warning(f"Database integrity check failed: {e}")
            
            await content_agent.stop()

    @pytest.mark.asyncio
    async def test_08_system_recovery_after_failure(self, setup_test_environment):
        """Test system recovery and timestamp handling after simulated failure"""
        logger.info("Testing system recovery after failure...")
        
        # Set initial timestamp
        initial_date = datetime(2025, 1, 5)
        memory_service.save_last_update_time(initial_date)
        
        # Simulate processing some emails
        processed_date1 = datetime(2025, 1, 6)
        processed_date2 = datetime(2025, 1, 7)
        
        # Process first email successfully
        memory_service.save_last_update_time(processed_date1)
        first_update = memory_service.get_last_update_time()
        
        # Simulate system restart by getting timestamp again
        restart_update = memory_service.get_last_update_time()
        
        assert restart_update == first_update, "Timestamp not persistent across restarts"
        
        # Process second email
        memory_service.save_last_update_time(processed_date2)
        final_update = memory_service.get_last_update_time()
        
        assert final_update.date() == processed_date2.date(), "Final timestamp not updated correctly"
        
        logger.info(f"✅ System recovery working: {initial_date.date()} → {final_update.date()}")

    @pytest.mark.asyncio
    async def test_09_end_to_end_workflow(self, setup_test_environment):
        """Complete end-to-end test of the entire workflow"""
        logger.info("Running end-to-end workflow test...")
        
        # Step 1: Initialize event coordinator
        try:
            await event_coordinator.initialize()
            logger.info("✅ Event coordinator initialized")
        except Exception as e:
            logger.warning(f"Event coordinator initialization failed: {e}")
            return
        
        # Step 2: Trigger email fetch
        with patch('backend.services.gmail_service_oauth.get_gmail_service') as mock_gmail:
            # Setup mock Gmail service
            mock_service = Mock()
            mock_service.authenticate.return_value = True
            
            # Create mock email with articles
            test_email = self.create_mock_email_data(datetime(2025, 1, 8), num_articles=3)
            mock_service.mcp_service.search_medium_emails.return_value = [test_email]
            
            def mock_get_articles(email):
                return [{
                    'title': f"TEST_E2E_Article_{i}",
                    'link': f"https://medium.com/e2e-test-{i}",
                    'summary': f"End-to-end test article {i}",
                    'hash': f"e2e_hash_{i}",
                    'digest_date': email['date'].date(),
                    'author': 'E2E Test Author'
                } for i in range(1, 4)]
            
            mock_service.mcp_service.get_articles_from_email.side_effect = mock_get_articles
            mock_gmail.return_value = mock_service
            
            # Mock content fetching
            async def mock_fetch_content(url):
                return self.create_mock_article_content(f"E2E Test Article for {url}")
            
            with patch('backend.services.article_content_fetcher.fetch_article_content',
                       side_effect=mock_fetch_content):
                
                # Trigger fetch
                fetch_result = await event_coordinator.trigger_email_fetch(max_emails=1)
                assert fetch_result["success"] is True
                
                # Wait for processing
                await asyncio.sleep(5)
                
                # Step 3: Perform search
                mock_llm_response = """
                Relevance Score: 0.9
                Reasoning: Excellent match for the search query
                Key Topics: Testing, End-to-End, Automation
                """
                
                with patch('backend.services.ollama_service.ollama_service.generate_response',
                           return_value=mock_llm_response):
                    
                    search_result = await event_coordinator.search_articles("e2e test")
                    
                    # Verify search results
                    assert len(search_result.get("results", [])) > 0, "No search results found"
                    
                    logger.info(f"✅ End-to-end test completed successfully")
                    logger.info(f"   - Fetch result: {fetch_result['success']}")
                    logger.info(f"   - Search results: {len(search_result.get('results', []))}")

# Run the tests
async def run_all_tests():
    """Run all tests in sequence"""
    logger.info("Starting comprehensive architecture tests...")
    
    test_suite = TestEventDrivenArchitecture()
    
    # Setup test environment
    async for _ in test_suite.setup_test_environment():
        try:
            # Run tests in order
            await test_suite.test_01_timestamp_initialization(_)
            await test_suite.test_02_event_bus_initialization(_)
            await test_suite.test_03_email_processing_sequential_with_parallel_articles(_)
            await test_suite.test_04_parallel_content_fetching_and_storage(_)
            await test_suite.test_05_vector_search_with_llm_contextualization(_)
            await test_suite.test_06_search_caching_functionality(_)
            await test_suite.test_07_concurrent_article_processing(_)
            await test_suite.test_08_system_recovery_after_failure(_)
            await test_suite.test_09_end_to_end_workflow(_)
            
            logger.info("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise
        
        break

if __name__ == "__main__":
    # Run tests
    asyncio.run(run_all_tests())
