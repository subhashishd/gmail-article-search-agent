#!/usr/bin/env python3
"""
Direct database insertion test to verify the schema and insertion process.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import hashlib

# Database config (same as in the backend)
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'postgres',
    'database': 'gmail_articles'
}

VECTOR_TABLE_NAME = 'medium_articles'

def test_direct_insertion():
    """Test direct insertion into the database."""
    
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        # Test article data
        test_article = {
            'title': 'Test Article for Database Verification',
            'link': 'https://test.com/test-article',
            'summary': 'This is a test article to verify database insertion works correctly.',
            'content': 'Full content of the test article for verification purposes.',
            'author': 'Test Author',
            'hash': hashlib.md5('test-article-unique'.encode()).hexdigest(),
            'digest_date': datetime.now().date()
        }
        
        # Create a simple embedding (just zeros for testing)
        embedding = [0.0] * 384
        
        print("🧪 Testing database insertion...")
        print(f"Table: {VECTOR_TABLE_NAME}")
        print(f"Article: {test_article['title']}")
        
        # Test the EXACT INSERT statement from our fixed code
        sql_statement = f"""
            INSERT INTO {VECTOR_TABLE_NAME} 
            (title, link, summary, content, author, hash, processed_at, embedding, digest_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        print("\n📝 SQL Statement:")
        print(sql_statement)
        
        print("\n🔍 Executing insertion...")
        cursor.execute(sql_statement, (
            test_article['title'][:500],
            test_article['link'][:1000],
            test_article['summary'][:2000],
            test_article['content'],
            test_article['author'][:200],
            test_article['hash'],
            datetime.now(),
            embedding,
            test_article['digest_date']
        ))
        
        conn.commit()
        print("✅ Insertion successful!")
        
        # Verify the insertion
        cursor.execute(f"SELECT COUNT(*) as count FROM {VECTOR_TABLE_NAME}")
        result = cursor.fetchone()
        print(f"📊 Total articles in database: {result['count']}")
        
        # Check the specific article
        cursor.execute(f"SELECT title, author FROM {VECTOR_TABLE_NAME} WHERE hash = %s", (test_article['hash'],))
        result = cursor.fetchone()
        if result:
            print(f"📄 Found inserted article: {result['title']} by {result['author']}")
        else:
            print("❌ Article not found after insertion!")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Database insertion failed: {e}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'pgcode'):
            print(f"PostgreSQL error code: {e.pgcode}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 DATABASE INSERTION VERIFICATION TEST")
    print("=" * 60)
    
    success = test_direct_insertion()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 DATABASE INSERTION TEST PASSED!")
    else:
        print("💥 DATABASE INSERTION TEST FAILED!")
    print("=" * 60)
