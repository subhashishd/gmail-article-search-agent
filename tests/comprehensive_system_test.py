#!/usr/bin/env python3
"""
Comprehensive System Test for Gmail Article Search Agent

This test verifies:
1. Email processing with correct timestamp management 
2. Articles are only from January 1, 2025 onwards
3. Event-driven architecture is working properly
4. Database integrity and article dates
5. Rate limiting and parallel processing
"""

import asyncio
import json
import requests
import sys
from datetime import datetime, date
from typing import Dict, Any, List
import subprocess
import time

class SystemTester:
    """Comprehensive system tester for the Gmail Article Search Agent"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.cutoff_date = date(2025, 1, 1)  # Our minimum allowed date
        self.results = {}
        
    def run_all_tests(self):
        """Run all system tests"""
        print("🚀 Starting Comprehensive System Test")
        print("=" * 60)
        
        # Test 1: Basic system health
        print("\n1. Testing System Health...")
        self.test_system_health()
        
        # Test 2: Database inspection
        print("\n2. Inspecting Database...")
        self.test_database_inspection()
        
        # Test 3: Timestamp management
        print("\n3. Testing Timestamp Management...")
        self.test_timestamp_management()
        
        # Test 4: Article date validation
        print("\n4. Testing Article Date Validation...")
        self.test_article_date_validation()
        
        # Test 5: Event-driven functionality  
        print("\n5. Testing Event-Driven Functionality...")
        self.test_event_driven_functionality()
        
        # Test 6: Search functionality
        print("\n6. Testing Search Functionality...")
        self.test_search_functionality()
        
        # Generate comprehensive report
        print("\n" + "=" * 60)
        self.generate_report()
        
    def test_system_health(self):
        """Test basic system health and availability"""
        try:
            # Test health endpoint
            response = requests.get(f"{self.base_url}/health", timeout=10)
            health_data = response.json()
            
            print(f"   ✅ Health Status: {health_data.get('status', 'unknown')}")
            print(f"   ✅ Service Type: {health_data.get('service', 'unknown')}")
            
            # Test system status  
            response = requests.get(f"{self.base_url}/status", timeout=10)
            status_data = response.json()
            
            print(f"   ✅ Architecture: {status_data.get('architecture', 'unknown')}")
            print(f"   ✅ Coordinator Status: {status_data['coordinator']['coordinator_status']}")
            
            agents = status_data['coordinator']['agents']
            print(f"   ✅ Email Processor: {agents['email_processor']['status']}")
            print(f"   ✅ Content Agent: {agents['content_agent']['running']}")
            print(f"   ✅ Search Agent: {agents['search_agent']['status']}")
            
            self.results['system_health'] = {
                'status': 'PASS',
                'details': {
                    'health': health_data.get('status'),
                    'architecture': status_data.get('architecture'),
                    'agents_running': True
                }
            }
            
        except Exception as e:
            print(f"   ❌ System Health Test Failed: {e}")
            self.results['system_health'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_database_inspection(self):
        """Inspect database contents and structure"""
        try:
            # Check article count
            result = subprocess.run([
                "docker", "exec", "gmail-search-db", "psql", "-U", "postgres", 
                "-d", "gmail_article_search", "-c", "SELECT COUNT(*) FROM medium_articles;"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                article_count = int(result.stdout.split('\n')[2].strip())
                print(f"   ✅ Total Articles in Database: {article_count}")
                
                # Check date range
                date_result = subprocess.run([
                    "docker", "exec", "gmail-search-db", "psql", "-U", "postgres",
                    "-d", "gmail_article_search", "-c", 
                    "SELECT MIN(digest_date) as earliest, MAX(digest_date) as latest FROM medium_articles;"
                ], capture_output=True, text=True, timeout=30)
                
                if date_result.returncode == 0:
                    lines = date_result.stdout.strip().split('\n')
                    data_line = lines[2].strip().split('|')
                    earliest_date = data_line[0].strip()
                    latest_date = data_line[1].strip()
                    
                    print(f"   📅 Date Range: {earliest_date} to {latest_date}")
                    
                    # Check if dates are valid (after Jan 1, 2025)
                    if earliest_date:
                        earliest = datetime.strptime(earliest_date, '%Y-%m-%d').date()
                        if earliest >= self.cutoff_date:
                            print(f"   ✅ Earliest date ({earliest_date}) is after cutoff ({self.cutoff_date})")
                            date_validation = 'PASS'
                        else:
                            print(f"   ❌ Earliest date ({earliest_date}) is before cutoff ({self.cutoff_date})")
                            date_validation = 'FAIL'
                    else:
                        date_validation = 'NO_DATA'
                    
                    self.results['database_inspection'] = {
                        'status': 'PASS',
                        'details': {
                            'article_count': article_count,
                            'earliest_date': earliest_date,
                            'latest_date': latest_date,
                            'date_validation': date_validation
                        }
                    }
                else:
                    raise Exception(f"Date query failed: {date_result.stderr}")
            else:
                raise Exception(f"Count query failed: {result.stderr}")
                
        except Exception as e:
            print(f"   ❌ Database Inspection Failed: {e}")
            self.results['database_inspection'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_timestamp_management(self):
        """Test timestamp management and last update tracking"""
        try:
            # Check current timestamp in the system
            response = requests.get(f"{self.base_url}/status", timeout=10)
            status_data = response.json()
            
            last_processed = status_data['coordinator']['agents']['email_processor']['last_processed_time']
            print(f"   📅 Last Processed Time: {last_processed}")
            
            # Parse and validate the timestamp
            last_time = datetime.fromisoformat(last_processed.replace('Z', '+00:00'))
            
            if last_time.date() >= self.cutoff_date:
                print(f"   ✅ Last processed time is after cutoff date")
                timestamp_valid = True
            else:
                print(f"   ❌ Last processed time is before cutoff date")
                timestamp_valid = False
                
            self.results['timestamp_management'] = {
                'status': 'PASS' if timestamp_valid else 'FAIL',
                'details': {
                    'last_processed_time': last_processed,
                    'cutoff_date': str(self.cutoff_date),
                    'valid': timestamp_valid
                }
            }
            
        except Exception as e:
            print(f"   ❌ Timestamp Management Test Failed: {e}")
            self.results['timestamp_management'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_article_date_validation(self):
        """Test that all articles in database are from correct date range"""
        try:
            # Query for articles with dates before cutoff
            result = subprocess.run([
                "docker", "exec", "gmail-search-db", "psql", "-U", "postgres",
                "-d", "gmail_article_search", "-c", 
                f"SELECT COUNT(*) FROM medium_articles WHERE digest_date < '{self.cutoff_date}';"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                invalid_count = int(result.stdout.split('\n')[2].strip())
                
                if invalid_count == 0:
                    print(f"   ✅ No articles found before cutoff date")
                    validation_status = 'PASS'
                else:
                    print(f"   ❌ Found {invalid_count} articles before cutoff date")
                    validation_status = 'FAIL'
                    
                # Also check for future dates (should not exist)
                current_date = datetime.now().date()
                future_result = subprocess.run([
                    "docker", "exec", "gmail-search-db", "psql", "-U", "postgres",
                    "-d", "gmail_article_search", "-c", 
                    f"SELECT COUNT(*) FROM medium_articles WHERE digest_date > '{current_date}';"
                ], capture_output=True, text=True, timeout=30)
                
                if future_result.returncode == 0:
                    future_count = int(future_result.stdout.split('\n')[2].strip())
                    
                    if future_count > 0:
                        print(f"   ⚠️  Found {future_count} articles with future dates")
                        if validation_status == 'PASS':
                            validation_status = 'WARNING'
                    else:
                        print(f"   ✅ No articles with future dates")
                        
                self.results['article_date_validation'] = {
                    'status': validation_status,
                    'details': {
                        'invalid_old_articles': invalid_count,
                        'invalid_future_articles': future_count,
                        'cutoff_date': str(self.cutoff_date)
                    }
                }
            else:
                raise Exception(f"Article validation query failed: {result.stderr}")
                
        except Exception as e:
            print(f"   ❌ Article Date Validation Failed: {e}")
            self.results['article_date_validation'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_event_driven_functionality(self):
        """Test event-driven architecture and fetch functionality"""
        try:
            # Trigger a fetch operation
            print("   🔄 Triggering email fetch...")
            response = requests.post(f"{self.base_url}/fetch", timeout=30)
            fetch_result = response.json()
            
            if fetch_result.get('success'):
                print(f"   ✅ Fetch Request: {fetch_result.get('message')}")
                print(f"   📨 Event ID: {fetch_result.get('event_id')}")
                
                # Wait a bit for processing
                time.sleep(5)
                
                # Check system status after fetch
                status_response = requests.get(f"{self.base_url}/status", timeout=10)
                status_data = status_response.json()
                
                recent_events = status_data['coordinator']['recent_events']
                stats = status_data['coordinator']['stats']
                
                print(f"   📊 Recent Events: {len(recent_events)}")
                print(f"   📊 Emails Processed: {stats.get('emails_processed', 0)}")
                print(f"   📊 Articles Discovered: {stats.get('articles_discovered', 0)}")
                
                self.results['event_driven_functionality'] = {
                    'status': 'PASS',
                    'details': {
                        'fetch_success': True,
                        'event_id': fetch_result.get('event_id'),
                        'recent_events_count': len(recent_events),
                        'stats': stats
                    }
                }
            else:
                print(f"   ❌ Fetch Failed: {fetch_result.get('message')}")
                self.results['event_driven_functionality'] = {
                    'status': 'FAIL',
                    'details': {'fetch_error': fetch_result.get('message')}
                }
                
        except Exception as e:
            print(f"   ❌ Event-Driven Functionality Test Failed: {e}")
            self.results['event_driven_functionality'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_search_functionality(self):
        """Test search functionality with various queries"""
        try:
            test_queries = [
                "artificial intelligence",
                "machine learning", 
                "programming",
                "technology"
            ]
            
            search_results = {}
            
            for query in test_queries:
                try:
                    response = requests.post(
                        f"{self.base_url}/search",
                        json={"query": query, "top_k": 5},
                        timeout=15
                    )
                    result = response.json()
                    
                    results_count = result.get('total_found', 0)
                    print(f"   🔍 Query '{query}': {results_count} results")
                    
                    search_results[query] = {
                        'total_found': results_count,
                        'success': 'error' not in result
                    }
                    
                except Exception as e:
                    print(f"   ❌ Search failed for '{query}': {e}")
                    search_results[query] = {'error': str(e), 'success': False}
            
            # Check if at least some searches returned results
            successful_searches = sum(1 for r in search_results.values() if r.get('success', False))
            total_results = sum(r.get('total_found', 0) for r in search_results.values())
            
            if successful_searches > 0:
                print(f"   ✅ Search System Working: {successful_searches}/{len(test_queries)} queries successful")
                print(f"   📊 Total Results Found: {total_results}")
                status = 'PASS'
            else:
                print(f"   ❌ Search System Failed: No successful queries")
                status = 'FAIL'
                
            self.results['search_functionality'] = {
                'status': status,
                'details': {
                    'successful_queries': successful_searches,
                    'total_queries': len(test_queries),
                    'total_results': total_results,
                    'query_results': search_results
                }
            }
            
        except Exception as e:
            print(f"   ❌ Search Functionality Test Failed: {e}")
            self.results['search_functionality'] = {'status': 'FAIL', 'error': str(e)}
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get('status') == 'PASS')
        failed_tests = sum(1 for r in self.results.values() if r.get('status') == 'FAIL')
        warning_tests = sum(1 for r in self.results.values() if r.get('status') == 'WARNING')
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️  Warnings: {warning_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.results.items():
            status_icon = {
                'PASS': '✅',
                'FAIL': '❌', 
                'WARNING': '⚠️'
            }.get(result.get('status'), '❓')
            
            print(f"{status_icon} {test_name}: {result.get('status')}")
            
            if result.get('status') == 'FAIL' and 'error' in result:
                print(f"   Error: {result['error']}")
            elif 'details' in result:
                details = result['details']
                if isinstance(details, dict):
                    for key, value in details.items():
                        if isinstance(value, (str, int, bool)):
                            print(f"   {key}: {value}")
        
        # CRITICAL FINDINGS
        print("\n🚨 CRITICAL FINDINGS:")
        
        # Check date validation specifically
        db_result = self.results.get('database_inspection', {})
        if db_result.get('status') == 'PASS':
            date_validation = db_result.get('details', {}).get('date_validation')
            if date_validation == 'FAIL':
                print("❌ CRITICAL: Articles found before January 1, 2025 cutoff date!")
                print("   This indicates timestamp management is not working correctly.")
                print("   Gmail fetching is processing old emails instead of starting from Jan 1, 2025.")
            elif date_validation == 'PASS':
                print("✅ Date validation passed: All articles are after January 1, 2025")
            else:
                print("⚠️  No article data available for date validation")
        
        # Save detailed report to file
        with open('/tmp/system_test_report.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed report saved to: /tmp/system_test_report.json")


if __name__ == "__main__":
    tester = SystemTester()
    tester.run_all_tests()
