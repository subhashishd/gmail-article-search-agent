"""
Rate Limiter for controlling API requests to external services like Medium.
Uses Redis for distributed rate limiting across multiple workers.
"""

import asyncio
import time
import logging
import os
from typing import Optional
import redis.asyncio as redis

class RateLimiter:
    """Redis-based rate limiter for API calls"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        self.logger = logging.getLogger("RateLimiter")
        
        # Rate limiting configs for different services
        self.limits = {
            "medium": {"requests": 6, "window": 60},   # 6 requests per minute (more conservative)
            "openai": {"requests": 60, "window": 60},  # 60 requests per minute
            "default": {"requests": 30, "window": 60}   # 30 requests per minute
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            self.logger.info(f"Rate limiter connected to Redis at {self.redis_url}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis for rate limiting: {e}")
            raise
    
    async def acquire(self, service: str = "default", timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.
        Returns True if allowed, False if rate limited.
        """
        if not self.redis_client:
            await self.initialize()
        
        config = self.limits.get(service, self.limits["default"])
        key = f"rate_limit:{service}"
        
        start_time = time.time()
        
        while True:
            try:
                # Get current count
                current = await self.redis_client.get(key)
                current_count = int(current) if current else 0
                
                if current_count < config["requests"]:
                    # We can make the request
                    pipe = self.redis_client.pipeline()
                    pipe.incr(key)
                    pipe.expire(key, config["window"])
                    await pipe.execute()
                    
                    self.logger.debug(f"Rate limit acquired for {service}: {current_count + 1}/{config['requests']}")
                    return True
                
                # Rate limited - check if we should wait or timeout
                if timeout is not None and (time.time() - start_time) >= timeout:
                    self.logger.warning(f"Rate limit timeout for {service}")
                    return False
                
                # Wait a bit before retrying
                ttl = await self.redis_client.ttl(key)
                wait_time = min(ttl if ttl > 0 else 1, 5)  # Wait at most 5 seconds
                
                self.logger.debug(f"Rate limited for {service}, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Error in rate limiter for {service}: {e}")
                # Fallback: allow request but log error
                return True
    
    async def get_status(self, service: str = "default") -> dict:
        """Get current rate limit status for a service"""
        if not self.redis_client:
            return {"error": "Redis not connected"}
        
        try:
            config = self.limits.get(service, self.limits["default"])
            key = f"rate_limit:{service}"
            
            current = await self.redis_client.get(key)
            current_count = int(current) if current else 0
            ttl = await self.redis_client.ttl(key)
            
            return {
                "service": service,
                "current_requests": current_count,
                "max_requests": config["requests"],
                "window_seconds": config["window"],
                "remaining": max(0, config["requests"] - current_count),
                "reset_in_seconds": ttl if ttl > 0 else 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.logger.info("Rate limiter cleanup complete")

# Context manager for easy usage
class RateLimitedRequest:
    """Context manager for rate-limited requests"""
    
    def __init__(self, rate_limiter: RateLimiter, service: str = "default", timeout: Optional[float] = None):
        self.rate_limiter = rate_limiter
        self.service = service
        self.timeout = timeout
        self.acquired = False
    
    async def __aenter__(self):
        self.acquired = await self.rate_limiter.acquire(self.service, self.timeout)
        if not self.acquired:
            raise Exception(f"Rate limit exceeded for {self.service}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Nothing to cleanup for now
        pass

# Global rate limiter instance - created lazily
rate_limiter = None

def get_rate_limiter():
    global rate_limiter
    if rate_limiter is None:
        rate_limiter = RateLimiter()
    return rate_limiter

# For backward compatibility
rate_limiter = get_rate_limiter()
