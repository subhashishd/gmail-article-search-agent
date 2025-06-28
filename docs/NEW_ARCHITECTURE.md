# New Event-Driven Architecture

## Overview

This document outlines the new event-driven architecture that addresses all the concurrency and processing concerns raised about Medium article processing.

## Core Architecture Changes

### 1. **Event-Driven Communication**
- **Redis Event Bus**: All agents communicate through Redis pub/sub
- **Async Processing**: Non-blocking operations across all components
- **Event Isolation**: Each agent handles its own failures without affecting others

### 2. **Proper Article Processing Flow**

```
Email Processing → Article Discovery → Content Fetching → Database Storage
     (Sequential)     (Parallel)        (Rate Limited)     (Concurrent Safe)
```

#### **Email Processor Agent**
- Processes emails **sequentially** to maintain chronological order
- Updates timestamp after each email (crash-safe)
- Publishes articles from each email **in parallel**
- Controls concurrency with semaphores

#### **Content Agent Workers**
- Process articles **in parallel** with controlled concurrency (5 workers default)
- Rate-limited content fetching from Medium (10 req/min)
- Database operations with proper conflict handling
- **NO LLM analysis during storage** (as per your requirement)

#### **Search Agent**
- Immediate search response from cached/vector results
- LLM enhancement **only during search** (not storage)
- Redis caching for frequent queries

## Concurrency Solutions

### 1. **Database Concurrency**
```sql
INSERT INTO articles (...) VALUES (...)
ON CONFLICT (hash) DO NOTHING
```
- Atomic operations with conflict resolution
- Individual connections per worker
- Retry logic with exponential backoff
- Graceful handling of duplicate articles

### 2. **Rate Limiting**
```python
async with RateLimitedRequest(rate_limiter, "medium", timeout=10.0):
    content = await fetch_article_content(url)
```
- Redis-based distributed rate limiting
- Per-service rate limits (Medium: 10/min, Default: 30/min)
- Automatic backoff and retry

### 3. **Article Processing Parallelism**
- **Email Level**: Sequential (maintains chronological order)
- **Article Level**: Parallel within each email (5 concurrent max)
- **Content Fetching**: Rate-limited parallel
- **Database Storage**: Concurrent safe with conflict resolution

## Key Benefits

### ✅ **Solves Your Original Concerns**

1. **Non-blocking Gmail Processing**: Email fetching doesn't block search operations
2. **Parallel Article Processing**: 10 articles from an email process in parallel
3. **Proper Timestamp Management**: Updates after each email processed
4. **No LLM During Storage**: Only vector embeddings stored, LLM used during search
5. **Crash Recovery**: Event-driven design survives system restarts

### ✅ **Performance Optimizations**

1. **Search Caching**: Frequent queries cached in Redis (5min TTL)
2. **Connection Pooling**: Proper database connection management
3. **Rate Limiting**: Respects Medium's API limits
4. **Controlled Concurrency**: Prevents system overload

### ✅ **Error Handling**

1. **Circuit Breakers**: Failed operations don't cascade
2. **Retry Logic**: Exponential backoff for transient failures
3. **Graceful Degradation**: System works even with component failures
4. **Monitoring**: Event streams provide excellent observability

## Component Responsibilities

### **Email Processor Agent**
- ✅ Sequential email processing
- ✅ Timestamp management after each email
- ✅ Parallel article publishing per email

### **Content Agent**
- ✅ Parallel article processing (5 workers)
- ✅ Rate-limited content fetching
- ✅ Database storage with embeddings
- ❌ **NO LLM analysis** (as per requirement)

### **Search Agent**
- ✅ Immediate vector search
- ✅ Optional LLM enhancement during search
- ✅ Redis caching for performance

### **Event Coordinator**
- ✅ Lightweight orchestration
- ✅ System monitoring and stats
- ✅ Health checks and status

## Configuration

### **Concurrency Controls**
- Email processing: Sequential (1 email at a time)
- Article processing: 5 workers per email
- Rate limiting: 10 Medium requests/minute
- Database connections: 1 per worker

### **Error Tolerance**
- Retry attempts: 3 per operation
- Rate limit timeout: 10 seconds
- Connection timeout: 30 seconds
- Target error rate: <1% (as requested)

## Docker Integration

### **New Services Added**
```yaml
redis:
  image: redis:7-alpine
  # For event bus and caching
```

### **Dependencies Updated**
```python
redis[hiredis]>=5.0.1
aiohttp>=3.9.1
```

## Migration Path

1. **Phase 1**: Deploy new architecture alongside existing (completed)
2. **Phase 2**: Test with small email batches
3. **Phase 3**: Switch frontend to use new endpoints
4. **Phase 4**: Remove old multi-agent coordinator

## API Endpoints

### **New Event-Driven Endpoints**
- `POST /search` - Event-driven search with caching
- `POST /fetch` - Trigger email processing
- `GET /status` - System status and metrics

### **Monitoring**
- Real-time event streams
- Agent health checks
- Processing statistics
- Error tracking

## Performance Expectations

- **Search Response**: 1-2 seconds (with LLM enhancement)
- **Parallel Processing**: 10 articles processed simultaneously
- **Error Rate**: <1% (with retry logic)
- **Cache Hit Rate**: >80% for frequent queries
- **Memory Usage**: Controlled with worker limits

This architecture directly addresses your requirements:
- ✅ Parallel article processing from emails
- ✅ Proper timestamp management
- ✅ No LLM during storage (only embeddings)
- ✅ LLM enhancement during search only
- ✅ Docker-ready for cloud migration
- ✅ Local-optimized with eventual cloud scaling
