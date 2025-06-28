# Queue Persistence Alternatives for Article Processing

## Current Problem
- **In-memory Python queue** loses all data on server restart
- **2,287 articles** would need complete reprocessing
- **No progress persistence** or recovery capability

## Alternative Solutions

### 1. Redis Queue (Recommended for Docker)
**Pros:**
- ✅ **Persistent** - Survives restarts
- ✅ **Fast** - In-memory but persistent
- ✅ **Docker-native** - Easy to add to docker-compose
- ✅ **Battle-tested** - Industry standard
- ✅ **Progress tracking** - Can store processing state

**Implementation:**
```python
import redis
import pickle

class RedisQueueService:
    def __init__(self):
        self.redis = redis.Redis(host='redis', port=6379, decode_responses=False)
        self.queue_key = 'article_processing_queue'
        self.progress_key = 'processing_progress'
    
    def add_articles(self, articles):
        for article in articles:
            self.redis.rpush(self.queue_key, pickle.dumps(article))
    
    def get_next_batch(self, batch_size=3):
        batch = []
        for _ in range(batch_size):
            item = self.redis.lpop(self.queue_key)
            if item:
                batch.append(pickle.loads(item))
        return batch
    
    def save_progress(self, progress):
        self.redis.set(self.progress_key, pickle.dumps(progress))
    
    def get_progress(self):
        data = self.redis.get(self.progress_key)
        return pickle.loads(data) if data else None
```

**Docker Addition:**
```yaml
services:
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
volumes:
  redis_data:
```

### 2. PostgreSQL Queue (Database-based)
**Pros:**
- ✅ **Already have PostgreSQL** - No new services
- ✅ **ACID transactions** - Guaranteed consistency
- ✅ **SQL queries** - Easy debugging and monitoring
- ✅ **Progress in DB** - Everything in one place

**Implementation:**
```sql
CREATE TABLE article_processing_queue (
    id SERIAL PRIMARY KEY,
    article_data JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    retries INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP
);

CREATE TABLE processing_progress (
    id INTEGER PRIMARY KEY DEFAULT 1,
    total_articles INTEGER,
    processed_articles INTEGER,
    successful_articles INTEGER,
    failed_articles INTEGER,
    status VARCHAR(50),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    current_batch INTEGER,
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 3. File-based Queue (Simple)
**Pros:**
- ✅ **Simple** - No new dependencies
- ✅ **Persistent** - Files survive restarts
- ✅ **Docker volumes** - Easy persistence

**Cons:**
- ⚠️ **File locking** - Concurrency issues
- ⚠️ **Performance** - Slower than memory/Redis

### 4. Celery + Redis/RabbitMQ (Full Solution)
**Pros:**
- ✅ **Professional** - Production-grade task queue
- ✅ **Distributed** - Can scale across servers
- ✅ **Monitoring** - Built-in web interface
- ✅ **Retry logic** - Advanced failure handling

**Cons:**
- ❌ **Complex** - Significant architecture change
- ❌ **Overkill** - For single-server setup

## Recommended Quick Fix: Database Queue

Since we already have PostgreSQL, the **fastest implementation** would be:

1. **Add queue tables** to existing database
2. **Modify ArticleQueueService** to use DB instead of memory
3. **Persist progress** in database table
4. **Resume processing** from last checkpoint on restart

## Implementation Priority:

1. **Quick Fix (2 hours)**: Database-based queue
2. **Better Solution (4 hours)**: Redis queue with Docker
3. **Enterprise Solution (2 days)**: Celery + Redis

Would you like me to implement the **database queue solution** first? It would:
- Use existing PostgreSQL container
- Add 2 simple tables
- Modify queue service to persist state
- Enable restart recovery
- Keep all 2,287 articles safe
