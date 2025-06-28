"""
Event Bus Implementation using Redis
Provides async event-driven communication between agents
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis.asyncio as redis
import uuid

@dataclass
class Event:
    """Event data structure"""
    id: str
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None

class EventBus:
    """Redis-based event bus for async agent communication"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        self.subscribers = {}
        self.logger = logging.getLogger("EventBus")
        self._listening = False
        self._listener_task = None
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {self.redis_url}")
            
            # Start listening for events
            await self._start_listener()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def publish(self, event_type: str, data: Dict[str, Any], source: str = "unknown", correlation_id: Optional[str] = None) -> str:
        """Publish an event to the bus"""
        event = Event(
            id=str(uuid.uuid4()),
            type=event_type,
            data=data,
            timestamp=datetime.now(),
            source=source,
            correlation_id=correlation_id
        )
        
        try:
            event_json = json.dumps({
                "id": event.id,
                "type": event.type,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "correlation_id": event.correlation_id
            })
            
            # Publish to Redis channel
            await self.redis_client.publish(f"events:{event_type}", event_json)
            
            # Store event for replay/debugging (with TTL)
            await self.redis_client.setex(
                f"event:{event.id}", 
                timedelta(hours=24), 
                event_json
            )
            
            self.logger.debug(f"Published event {event.id}: {event_type}")
            return event.id
            
        except Exception as e:
            self.logger.error(f"Failed to publish event {event_type}: {e}")
            raise
    
    async def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        """Subscribe to events of a specific type"""
        is_new_event_type = event_type not in self.subscribers
        
        if is_new_event_type:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(handler)
        self.logger.info(f"Subscribed to event type: {event_type}")
        
        # If this is a new event type and we're already listening, 
        # we need to subscribe to the Redis channel
        if is_new_event_type and self._listening and hasattr(self, '_pubsub'):
            try:
                await self._pubsub.subscribe(f"events:{event_type}")
                self.logger.info(f"Subscribed to Redis channel: events:{event_type}")
            except Exception as e:
                self.logger.error(f"Failed to subscribe to Redis channel events:{event_type}: {e}")
    
    async def _start_listener(self):
        """Start listening for Redis pub/sub events"""
        if self._listening:
            return
            
        self._listening = True
        self._listener_task = asyncio.create_task(self._listen_loop())
        self.logger.info("Started event listener")
    
    async def _listen_loop(self):
        """Main listening loop for Redis pub/sub"""
        try:
            pubsub = self.redis_client.pubsub()
            self._pubsub = pubsub  # Store reference for dynamic subscriptions
            
            # Subscribe to all event channels we have handlers for
            for event_type in self.subscribers.keys():
                await pubsub.subscribe(f"events:{event_type}")
                self.logger.info(f"Initial subscription to Redis channel: events:{event_type}")
            
            # If no initial subscriptions, subscribe to a dummy channel to keep listener alive
            if not self.subscribers:
                await pubsub.subscribe("events:__keepalive__")
                self.logger.info("No initial subscriptions, subscribed to keepalive channel")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    # Skip keepalive messages
                    if not message['channel'].endswith('__keepalive__'):
                        await self._handle_message(message)
                    
        except Exception as e:
            self.logger.error(f"Error in event listener: {e}")
            if self._listening:
                # Restart listener after delay
                await asyncio.sleep(5)
                self._listener_task = asyncio.create_task(self._listen_loop())
    
    async def _handle_message(self, message):
        """Handle incoming Redis message"""
        try:
            # Parse event from message
            event_data = json.loads(message['data'])
            event = Event(
                id=event_data['id'],
                type=event_data['type'],
                data=event_data['data'],
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                source=event_data['source'],
                correlation_id=event_data.get('correlation_id')
            )
            
            # Find handlers for this event type
            if event.type in self.subscribers:
                for handler in self.subscribers[event.type]:
                    try:
                        # Run handler in background to avoid blocking
                        asyncio.create_task(self._run_handler(handler, event))
                    except Exception as e:
                        self.logger.error(f"Error calling handler for {event.type}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def _run_handler(self, handler: Callable, event: Event):
        """Run event handler with error isolation"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            self.logger.error(f"Handler error for event {event.id}: {e}")
    
    async def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get recent event history for debugging"""
        try:
            # Get event keys (limited by TTL)
            keys = await self.redis_client.keys("event:*")
            events = []
            
            for key in keys[-limit:]:  # Get latest events
                event_json = await self.redis_client.get(key)
                if event_json:
                    event_data = json.loads(event_json)
                    if event_type is None or event_data['type'] == event_type:
                        events.append(Event(
                            id=event_data['id'],
                            type=event_data['type'],
                            data=event_data['data'],
                            timestamp=datetime.fromisoformat(event_data['timestamp']),
                            source=event_data['source'],
                            correlation_id=event_data.get('correlation_id')
                        ))
            
            return sorted(events, key=lambda e: e.timestamp, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting event history: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup Redis connections"""
        self._listening = False
        
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("Event bus cleanup complete")

# Global event bus instance - created lazily
event_bus = None

def get_event_bus():
    global event_bus
    if event_bus is None:
        event_bus = EventBus()
    return event_bus

# For backward compatibility
event_bus = get_event_bus()
