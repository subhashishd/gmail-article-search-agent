"""
Shared Memory Manager for Agent System

Provides:
- Episodic Memory: Agent experiences and outcomes over time
- Semantic Memory: Factual knowledge and learned patterns  
- Vector Search: Retrieve relevant past experiences
- Cross-Agent Memory Sharing: Shared knowledge base
"""

import asyncio
import json
import logging
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis
import psycopg2
from psycopg2.extras import RealDictCursor
from backend.config import config


@dataclass
class EpisodicMemory:
    """Individual agent experience/episode"""
    id: str
    agent_id: str
    task_type: str
    context: Dict[str, Any]
    actions_taken: List[Dict[str, Any]]
    outcome: Dict[str, Any]
    success: bool
    learned_patterns: List[str]
    timestamp: datetime
    embedding: Optional[List[float]] = None


@dataclass 
class SemanticMemory:
    """Factual knowledge and learned patterns"""
    id: str
    knowledge_type: str  # "pattern", "fact", "rule", "strategy"
    content: str
    confidence_score: float
    supporting_episodes: List[str]  # Episode IDs that support this knowledge
    created_by: str  # Agent ID
    last_validated: datetime
    usage_count: int
    embedding: Optional[List[float]] = None


class MemoryManager:
    """Centralized memory management for all agents"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        self.logger = logging.getLogger("MemoryManager")
        
        # Initialize embedding model for semantic search
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Memory caches
        self.episodic_cache = {}
        self.semantic_cache = {}
        
    async def initialize(self):
        """Initialize memory manager"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Create database tables if they don't exist
            await self._create_memory_tables()
            
            self.logger.info("Memory Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Memory Manager: {e}")
            raise
    
    async def _create_memory_tables(self):
        """Create memory tables in database"""
        try:
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME
            )
            
            with conn.cursor() as cursor:
                # Episodic memory table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS episodic_memory (
                        id VARCHAR(255) PRIMARY KEY,
                        agent_id VARCHAR(255) NOT NULL,
                        task_type VARCHAR(255) NOT NULL,
                        context JSONB NOT NULL,
                        actions_taken JSONB NOT NULL,
                        outcome JSONB NOT NULL,
                        success BOOLEAN NOT NULL,
                        learned_patterns JSONB NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        embedding vector(384),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Semantic memory table  
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_memory (
                        id VARCHAR(255) PRIMARY KEY,
                        knowledge_type VARCHAR(100) NOT NULL,
                        content TEXT NOT NULL,
                        confidence_score FLOAT NOT NULL,
                        supporting_episodes JSONB NOT NULL,
                        created_by VARCHAR(255) NOT NULL,
                        last_validated TIMESTAMP NOT NULL,
                        usage_count INTEGER DEFAULT 0,
                        embedding vector(384),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_agent_id ON episodic_memory(agent_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_task_type ON episodic_memory(task_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic_memory(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_memory(knowledge_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_confidence ON semantic_memory(confidence_score)")
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create memory tables: {e}")
            raise
    
    # EPISODIC MEMORY METHODS
    
    async def store_episode(self, 
                          agent_id: str,
                          task_type: str, 
                          context: Dict[str, Any],
                          actions_taken: List[Dict[str, Any]],
                          outcome: Dict[str, Any],
                          success: bool,
                          learned_patterns: List[str] = None) -> str:
        """Store a new episodic memory"""
        
        episode_id = str(uuid.uuid4())
        learned_patterns = learned_patterns or []
        
        # Create episode object
        episode = EpisodicMemory(
            id=episode_id,
            agent_id=agent_id,
            task_type=task_type,
            context=context,
            actions_taken=actions_taken,
            outcome=outcome,
            success=success,
            learned_patterns=learned_patterns,
            timestamp=datetime.now()
        )
        
        # Generate embedding for semantic search
        episode_text = f"{task_type} {json.dumps(context)} {json.dumps(outcome)} {' '.join(learned_patterns)}"
        episode.embedding = self.embedding_model.encode(episode_text).tolist()
        
        try:
            # Store in database
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME
            )
            
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO episodic_memory 
                    (id, agent_id, task_type, context, actions_taken, outcome, 
                     success, learned_patterns, timestamp, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    episode.id, episode.agent_id, episode.task_type,
                    json.dumps(episode.context), json.dumps(episode.actions_taken),
                    json.dumps(episode.outcome), episode.success,
                    json.dumps(episode.learned_patterns), episode.timestamp,
                    episode.embedding
                ))
            
            conn.commit()
            conn.close()
            
            # Cache in Redis for fast access
            await self.redis_client.setex(
                f"episode:{episode_id}",
                3600,  # 1 hour TTL
                json.dumps(asdict(episode), default=str)
            )
            
            self.logger.info(f"Stored episode {episode_id} for agent {agent_id}")
            
            # Extract semantic knowledge from successful episodes
            if success and learned_patterns:
                await self._extract_semantic_knowledge(episode)
            
            return episode_id
            
        except Exception as e:
            self.logger.error(f"Failed to store episode: {e}")
            raise
    
    async def retrieve_similar_episodes(self, 
                                      query: str,
                                      agent_id: Optional[str] = None,
                                      task_type: Optional[str] = None,
                                      success_only: bool = False,
                                      limit: int = 5) -> List[EpisodicMemory]:
        """Retrieve episodic memories similar to query using vector search"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        try:
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME,
                cursor_factory=RealDictCursor
            )
            
            # Build query with filters
            where_conditions = ["embedding IS NOT NULL"]
            params = [query_embedding, limit]
            
            if agent_id:
                where_conditions.append("agent_id = %s")
                params.insert(-1, agent_id)
            
            if task_type:
                where_conditions.append("task_type = %s")
                params.insert(-1, task_type)
                
            if success_only:
                where_conditions.append("success = true")
            
            where_clause = " AND ".join(where_conditions)
            
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT *, (embedding <=> %s) as similarity
                    FROM episodic_memory
                    WHERE {where_clause}
                    ORDER BY similarity
                    LIMIT %s
                """, params)
                
                results = cursor.fetchall()
            
            conn.close()
            
            # Convert to EpisodicMemory objects
            episodes = []
            for row in results:
                episode = EpisodicMemory(
                    id=row['id'],
                    agent_id=row['agent_id'],
                    task_type=row['task_type'],
                    context=row['context'],
                    actions_taken=row['actions_taken'],
                    outcome=row['outcome'],
                    success=row['success'],
                    learned_patterns=row['learned_patterns'],
                    timestamp=row['timestamp'],
                    embedding=row['embedding']
                )
                episodes.append(episode)
            
            self.logger.info(f"Retrieved {len(episodes)} similar episodes for query: {query}")
            return episodes
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve similar episodes: {e}")
            return []
    
    # SEMANTIC MEMORY METHODS
    
    async def store_semantic_knowledge(self,
                                     knowledge_type: str,
                                     content: str,
                                     confidence_score: float,
                                     created_by: str,
                                     supporting_episodes: List[str] = None) -> str:
        """Store semantic knowledge"""
        
        knowledge_id = str(uuid.uuid4())
        supporting_episodes = supporting_episodes or []
        
        knowledge = SemanticMemory(
            id=knowledge_id,
            knowledge_type=knowledge_type,
            content=content,
            confidence_score=confidence_score,
            supporting_episodes=supporting_episodes,
            created_by=created_by,
            last_validated=datetime.now(),
            usage_count=0
        )
        
        # Generate embedding
        knowledge.embedding = self.embedding_model.encode(content).tolist()
        
        try:
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME
            )
            
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO semantic_memory
                    (id, knowledge_type, content, confidence_score, supporting_episodes,
                     created_by, last_validated, usage_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    knowledge.id, knowledge.knowledge_type, knowledge.content,
                    knowledge.confidence_score, json.dumps(knowledge.supporting_episodes),
                    knowledge.created_by, knowledge.last_validated, 
                    knowledge.usage_count, knowledge.embedding
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored semantic knowledge {knowledge_id}: {knowledge_type}")
            return knowledge_id
            
        except Exception as e:
            self.logger.error(f"Failed to store semantic knowledge: {e}")
            raise
    
    async def retrieve_relevant_knowledge(self,
                                        query: str,
                                        knowledge_type: Optional[str] = None,
                                        min_confidence: float = 0.5,
                                        limit: int = 5) -> List[SemanticMemory]:
        """Retrieve relevant semantic knowledge"""
        
        query_embedding = self.embedding_model.encode(query).tolist()
        
        try:
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME,
                cursor_factory=RealDictCursor
            )
            
            where_conditions = [
                "embedding IS NOT NULL",
                "confidence_score >= %s"
            ]
            params = [query_embedding, min_confidence, limit]
            
            if knowledge_type:
                where_conditions.append("knowledge_type = %s")
                params.insert(-1, knowledge_type)
            
            where_clause = " AND ".join(where_conditions)
            
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT *, (embedding <=> %s) as similarity
                    FROM semantic_memory
                    WHERE {where_clause}
                    ORDER BY similarity, confidence_score DESC
                    LIMIT %s
                """, params)
                
                results = cursor.fetchall()
            
            conn.close()
            
            # Convert and update usage count
            knowledge_items = []
            for row in results:
                knowledge = SemanticMemory(
                    id=row['id'],
                    knowledge_type=row['knowledge_type'],
                    content=row['content'],
                    confidence_score=row['confidence_score'],
                    supporting_episodes=row['supporting_episodes'],
                    created_by=row['created_by'],
                    last_validated=row['last_validated'],
                    usage_count=row['usage_count'],
                    embedding=row['embedding']
                )
                knowledge_items.append(knowledge)
                
                # Increment usage count (async)
                asyncio.create_task(self._increment_usage_count(knowledge.id))
            
            return knowledge_items
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve semantic knowledge: {e}")
            return []
    
    async def _extract_semantic_knowledge(self, episode: EpisodicMemory):
        """Extract semantic knowledge from successful episodes"""
        
        if not episode.success or not episode.learned_patterns:
            return
        
        for pattern in episode.learned_patterns:
            # Check if similar knowledge already exists
            existing = await self.retrieve_relevant_knowledge(
                pattern, 
                knowledge_type="pattern",
                limit=1
            )
            
            if existing and existing[0].content.lower() == pattern.lower():
                # Update confidence of existing knowledge
                await self._update_knowledge_confidence(existing[0].id, 0.1)
            else:
                # Create new semantic knowledge
                await self.store_semantic_knowledge(
                    knowledge_type="pattern",
                    content=pattern,
                    confidence_score=0.7,  # Initial confidence
                    created_by=episode.agent_id,
                    supporting_episodes=[episode.id]
                )
    
    async def _increment_usage_count(self, knowledge_id: str):
        """Increment usage count for knowledge item"""
        try:
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME
            )
            
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE semantic_memory 
                    SET usage_count = usage_count + 1 
                    WHERE id = %s
                """, (knowledge_id,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to increment usage count: {e}")
    
    async def _update_knowledge_confidence(self, knowledge_id: str, confidence_boost: float):
        """Update confidence score for existing knowledge"""
        try:
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME
            )
            
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE semantic_memory 
                    SET confidence_score = LEAST(confidence_score + %s, 1.0),
                        last_validated = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (confidence_boost, knowledge_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to update knowledge confidence: {e}")
    
    # UTILITY METHODS
    
    async def get_agent_memory_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get memory summary for an agent"""
        try:
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME,
                cursor_factory=RealDictCursor
            )
            
            with conn.cursor() as cursor:
                # Episode statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_episodes,
                        COUNT(*) FILTER (WHERE success = true) as successful_episodes,
                        COUNT(DISTINCT task_type) as unique_task_types
                    FROM episodic_memory 
                    WHERE agent_id = %s
                """, (agent_id,))
                
                episode_stats = cursor.fetchone()
                
                # Knowledge contributions
                cursor.execute("""
                    SELECT COUNT(*) as knowledge_contributed
                    FROM semantic_memory
                    WHERE created_by = %s
                """, (agent_id,))
                
                knowledge_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                "agent_id": agent_id,
                "episodic_memory": {
                    "total_episodes": episode_stats['total_episodes'],
                    "successful_episodes": episode_stats['successful_episodes'],
                    "success_rate": episode_stats['successful_episodes'] / max(episode_stats['total_episodes'], 1),
                    "unique_task_types": episode_stats['unique_task_types']
                },
                "semantic_contributions": knowledge_stats['knowledge_contributed']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory summary: {e}")
            return {"agent_id": agent_id, "error": str(e)}
    
    async def cleanup(self):
        """Cleanup memory manager resources"""
        if self.redis_client:
            await self.redis_client.close()
        self.logger.info("Memory Manager cleanup complete")


# Global memory manager instance
memory_manager = MemoryManager()
