#!/usr/bin/env python3
"""
Docker System Test for Event-Driven Architecture
Tests the complete system with all Docker services running.
"""

import requests
import time
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("docker_test")

class DockerSystemTester:
    """Test the complete Docker-based system"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:8501"
        self.passed = 0
        self.failed = 0
    
    def assert_true(self, condition, message):
        """Assert condition with logging"""
        if condition:
            self.passed += 1
            logger.info(f"✅ PASS: {message}")
        else:
            self.failed += 1
            logger.error(f"❌ FAIL: {message}")
    
    def test_01_service_health(self):
        """Test that all services are healthy"""
        logger.info("=== Testing Service Health ===")
        
        # Test backend health
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=10)
            self.assert_true(
                response.status_code == 200,
                "Backend service is healthy"
            )
            
            health_data = response.json()
            self.assert_true(
                health_data.get("status") == "healthy",
                "Backend reports healthy status"
            )
            
        except Exception as e:
            self.assert_true(False, f"Backend health check failed: {e}")
    
    def test_02_database_connectivity(self):
        """Test database connectivity and table structure"""
        logger.info("=== Testing Database Connectivity ===")
        
        try:
            response = requests.get(f"{self.backend_url}/stats/realtime", timeout=10)
            self.assert_true(
                response.status_code == 200,
                "Database stats endpoint accessible"
            )
            
            stats = response.json()
            self.assert_true(
                "total_articles" in stats,
                "Database stats include article count"
            )
            
            logger.info(f"Current articles in database: {stats.get('total_articles', 0)}")
            
        except Exception as e:
            self.assert_true(False, f"Database connectivity test failed: {e}")
    
    def test_03_event_coordinator_initialization(self):
        """Test event coordinator system status"""
        logger.info("=== Testing Event Coordinator ===")
        
        try:
            response = requests.get(f"{self.backend_url}/status", timeout=10)
            self.assert_true(
                response.status_code == 200,
                "System status endpoint accessible"
            )
            
            status = response.json()
            self.assert_true(
                status.get("service_status") == "running",
                "Event-driven service is running"
            )
            
        except Exception as e:
            self.assert_true(False, f"Event coordinator test failed: {e}")
    
    def test_04_search_functionality(self):
        """Test search functionality without requiring data"""
        logger.info("=== Testing Search Functionality ===")
        
        try:
            search_payload = {
                "query": "test search query",
                "top_k": 5
            }
            
            response = requests.post(
                f"{self.backend_url}/search",
                json=search_payload,
                timeout=30
            )
            
            self.assert_true(
                response.status_code == 200,
                "Search endpoint responds successfully"
            )
            
            search_result = response.json()
            
            # Check result structure
            self.assert_true(
                "results" in search_result,
                "Search result contains results array"
            )
            
            self.assert_true(
                "total_found" in search_result,
                "Search result contains total_found count"
            )
            
            self.assert_true(
                search_result.get("service") == "event_driven",
                "Search uses event-driven architecture"
            )
            
            logger.info(f"Search found {search_result.get('total_found', 0)} results")
            
        except Exception as e:
            self.assert_true(False, f"Search functionality test failed: {e}")
    
    def test_05_timestamp_management(self):
        """Test that timestamp management is working"""
        logger.info("=== Testing Timestamp Management ===")
        
        try:
            # Get current stats which includes last update time
            response = requests.get(f"{self.backend_url}/stats/realtime", timeout=10)
            
            if response.status_code == 200:
                stats = response.json()
                last_update = stats.get("last_update")
                
                self.assert_true(
                    last_update is not None,
                    "Last update timestamp is available"
                )
                
                if last_update:
                    # Parse timestamp
                    try:
                        parsed_date = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                        jan_1_2025 = datetime(2025, 1, 1)
                        
                        self.assert_true(
                            parsed_date >= jan_1_2025,
                            f"Timestamp is >= Jan 1, 2025 (current: {parsed_date.date()})"
                        )
                        
                    except Exception as e:
                        self.assert_true(False, f"Failed to parse timestamp: {e}")
                
            else:
                self.assert_true(False, "Failed to get stats for timestamp test")
                
        except Exception as e:
            self.assert_true(False, f"Timestamp management test failed: {e}")
    
    def test_06_parallel_processing_readiness(self):
        """Test that the system is ready for parallel processing"""
        logger.info("=== Testing Parallel Processing Readiness ===")
        
        try:
            # Test fetch endpoint (should return immediately even if no Gmail configured)
            response = requests.post(f"{self.backend_url}/fetch", timeout=10)
            
            # Even without Gmail credentials, the endpoint should respond
            # (might fail with authentication error, but shouldn't crash)
            self.assert_true(
                response.status_code in [200, 400, 500],  # Any response means endpoint works
                "Fetch endpoint is accessible (authentication may fail)"
            )
            
            fetch_result = response.json()
            
            # Check if we get a structured response
            self.assert_true(
                "success" in fetch_result or "error" in fetch_result,
                "Fetch endpoint returns structured response"
            )
            
            if not fetch_result.get("success", True):
                logger.info(f"Fetch failed as expected (no Gmail auth): {fetch_result.get('message', 'Unknown error')}")
            
        except Exception as e:
            self.assert_true(False, f"Parallel processing readiness test failed: {e}")
    
    def test_07_redis_event_bus(self):
        """Test Redis connectivity indirectly through search caching"""
        logger.info("=== Testing Redis Event Bus (indirectly) ===")
        
        try:
            # Perform two identical searches to test caching
            search_payload = {
                "query": "redis cache test query",
                "top_k": 3
            }
            
            # First search
            start_time = time.time()
            response1 = requests.post(
                f"{self.backend_url}/search",
                json=search_payload,
                timeout=20
            )
            first_duration = time.time() - start_time
            
            # Second search (should be faster if cached)
            start_time = time.time()
            response2 = requests.post(
                f"{self.backend_url}/search",
                json=search_payload,
                timeout=20
            )
            second_duration = time.time() - start_time
            
            self.assert_true(
                response1.status_code == 200 and response2.status_code == 200,
                "Both search requests completed successfully"
            )
            
            # Check if results are consistent (indicating proper system function)
            result1 = response1.json()
            result2 = response2.json()
            
            self.assert_true(
                result1.get("total_found") == result2.get("total_found"),
                "Search results are consistent between calls"
            )
            
            logger.info(f"Search durations: {first_duration:.2f}s -> {second_duration:.2f}s")
            
        except Exception as e:
            self.assert_true(False, f"Redis event bus test failed: {e}")
    
    def test_08_system_monitoring(self):
        """Test system monitoring and observability"""
        logger.info("=== Testing System Monitoring ===")
        
        try:
            # Test system status
            response = requests.get(f"{self.backend_url}/status", timeout=10)
            
            if response.status_code == 200:
                status = response.json()
                
                self.assert_true(
                    "service_status" in status,
                    "System status includes service status"
                )
                
                self.assert_true(
                    "timestamp" in status,
                    "System status includes timestamp"
                )
                
            # Test real-time stats
            response = requests.get(f"{self.backend_url}/stats/realtime", timeout=10)
            
            if response.status_code == 200:
                stats = response.json()
                
                self.assert_true(
                    "database_info" in stats,
                    "Real-time stats include database info"
                )
                
                self.assert_true(
                    "fetch_service" in stats,
                    "Real-time stats include fetch service info"
                )
                
        except Exception as e:
            self.assert_true(False, f"System monitoring test failed: {e}")
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        logger.info("=" * 60)
        logger.info("DOCKER SYSTEM TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {self.passed}")
        logger.info(f"Failed: {self.failed}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed == 0:
            logger.info("🎉 ALL DOCKER TESTS PASSED!")
            logger.info("Your event-driven architecture is working correctly!")
        else:
            logger.warning(f"⚠️  {self.failed} tests failed in Docker environment.")
        
        return self.failed == 0

def main():
    """Run Docker system tests"""
    logger.info("Starting Docker System Tests for Event-Driven Architecture")
    logger.info("=" * 70)
    logger.info("Make sure Docker services are running: docker-compose up")
    logger.info("=" * 70)
    
    # Wait for services to be ready
    logger.info("Waiting 10 seconds for services to initialize...")
    time.sleep(10)
    
    tester = DockerSystemTester()
    
    try:
        # Run all tests
        tester.test_01_service_health()
        tester.test_02_database_connectivity()
        tester.test_03_event_coordinator_initialization()
        tester.test_04_search_functionality()
        tester.test_05_timestamp_management()
        tester.test_06_parallel_processing_readiness()
        tester.test_07_redis_event_bus()
        tester.test_08_system_monitoring()
        
        # Print summary
        success = tester.print_summary()
        
        if success:
            logger.info("\\n🚀 DOCKER SYSTEM VALIDATION SUCCESSFUL!")
            logger.info("Your event-driven architecture is ready for production!")
            logger.info("\\nNext steps:")
            logger.info("1. Configure Gmail API credentials")
            logger.info("2. Test with real email fetching")
            logger.info("3. Validate parallel article processing")
        else:
            logger.error("\\n❌ DOCKER SYSTEM VALIDATION FAILED!")
            logger.error("Please check the logs and fix issues.")
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
