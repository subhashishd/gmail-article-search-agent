"""
Base Agent Class for Multi-Agent Framework

Implements core agent interface following LangChain conventions with:
- Structured message handling
- Tool integration capabilities  
- Memory management
- Observability and logging
- Async/await patterns for scalability
"""

import asyncio
import logging
import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# Agent Message Types
class MessageType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    COMMAND = "command"
    EVENT = "event"
    ERROR = "error"

@dataclass
class AgentMessage:
    """Standardized message format for agent communication."""
    id: str
    agent_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    conversation_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "conversation_id": self.conversation_id,
            "parent_message_id": self.parent_message_id,
            "metadata": self.metadata or {}
        }

@dataclass
class AgentResponse:
    """Standardized response format from agents."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata or {},
            "execution_time": self.execution_time
        }

class AgentCapability:
    """Represents a capability/tool that an agent can use."""
    
    def __init__(self, name: str, description: str, function: Callable, schema: Optional[Dict] = None):
        self.name = name
        self.description = description
        self.function = function
        self.schema = schema or {}
    
    async def execute(self, **kwargs) -> Any:
        """Execute the capability with given parameters."""
        if asyncio.iscoroutinefunction(self.function):
            return await self.function(**kwargs)
        else:
            return self.function(**kwargs)

class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.
    
    Provides core functionality:
    - Message handling and routing
    - Tool/capability management
    - Memory and state management
    - Logging and observability
    - Error handling and recovery
    """
    
    def __init__(self, 
                 agent_id: str,
                 name: str,
                 description: str,
                 capabilities: Optional[List[AgentCapability]] = None):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.memory = {}
        self.conversation_history = []
        self.logger = self._setup_logger()
        self.is_active = False
        self.message_handlers = {}
        
        # Register default message handlers
        self._register_handlers()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup agent-specific logger."""
        logger = logging.getLogger(f"agent.{self.agent_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - [{self.name}] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _register_handlers(self):
        """Register default message handlers."""
        self.message_handlers = {
            MessageType.QUERY: self.handle_query,
            MessageType.COMMAND: self.handle_command,
            MessageType.EVENT: self.handle_event,
        }
    
    def add_capability(self, capability: AgentCapability):
        """Add a new capability to the agent."""
        self.capabilities.append(capability)
        self.logger.info(f"Added capability: {capability.name}")
    
    def get_capability(self, name: str) -> Optional[AgentCapability]:
        """Get a capability by name."""
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None
    
    async def start(self):
        """Start the agent and initialize resources."""
        self.logger.info(f"Starting agent {self.name}")
        self.is_active = True
        await self.initialize()
    
    async def stop(self):
        """Stop the agent and cleanup resources."""
        self.logger.info(f"Stopping agent {self.name}")
        self.is_active = False
        await self.cleanup()
    
    @abstractmethod
    async def initialize(self):
        """Initialize agent-specific resources."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup agent-specific resources."""
        pass
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Process an incoming message and generate a response.
        
        Args:
            message: The message to process
            
        Returns:
            AgentResponse with result or error
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing message {message.id} of type {message.message_type.value}")
            
            # Add to conversation history
            self.conversation_history.append(message)
            
            # Get appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if not handler:
                raise ValueError(f"No handler for message type: {message.message_type}")
            
            # Process the message
            result = await handler(message)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            response = AgentResponse(
                success=True,
                data=result,
                execution_time=execution_time,
                metadata={"agent_id": self.agent_id, "message_id": message.id}
            )
            
            self.logger.info(f"Successfully processed message {message.id} in {execution_time:.3f}s")
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Error processing message {message.id}: {str(e)}"
            
            self.logger.error(error_msg)
            
            return AgentResponse(
                success=False,
                error=error_msg,
                execution_time=execution_time,
                metadata={"agent_id": self.agent_id, "message_id": message.id}
            )
    
    @abstractmethod
    async def handle_query(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle a query message."""
        pass
    
    async def handle_command(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle a command message."""
        command = message.content.get("command")
        if command == "status":
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "is_active": self.is_active,
                "capabilities": [cap.name for cap in self.capabilities],
                "memory_keys": list(self.memory.keys())
            }
        elif command == "reset":
            self.memory.clear()
            self.conversation_history.clear()
            return {"message": "Agent state reset successfully"}
        else:
            raise ValueError(f"Unknown command: {command}")
    
    async def handle_event(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle an event message."""
        event_type = message.content.get("event_type")
        self.logger.info(f"Received event: {event_type}")
        return {"acknowledged": True, "event_type": event_type}
    
    def create_message(self, 
                      message_type: MessageType, 
                      content: Dict[str, Any],
                      conversation_id: Optional[str] = None) -> AgentMessage:
        """Create a new message from this agent."""
        return AgentMessage(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            conversation_id=conversation_id
        )
    
    def set_memory(self, key: str, value: Any):
        """Store a value in agent memory."""
        self.memory[key] = value
        self.logger.debug(f"Set memory: {key}")
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from agent memory."""
        return self.memory.get(key, default)
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[AgentMessage]:
        """Get conversation history with optional limit."""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history.copy()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "schema": cap.schema
                } for cap in self.capabilities
            ],
            "memory_size": len(self.memory),
            "conversation_length": len(self.conversation_history)
        }
