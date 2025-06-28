"""
RAG Evaluation Service using RAGAS Framework

This service provides comprehensive evaluation of the RAG system quality including:
- Faithfulness: LLM responses grounded in retrieved context
- Answer Relevancy: Response relevance to user query  
- Context Precision: Quality of retrieved chunks ranking
- Context Recall: Completeness of retrieval
- Answer Semantic Similarity: Response quality metrics
- Custom metrics for enterprise monitoring

Integrates with monitoring infrastructure for continuous evaluation.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Core evaluation imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_semantic_similarity,
        answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not available, falling back to lightweight evaluation")

# Lightweight evaluation alternatives
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

from backend.monitoring import (
    monitor_agent_operation,
    record_article_processing,
    llm_requests_total,
    agent_operations_total
)

class EvaluationMode(Enum):
    """Evaluation modes for different scenarios."""
    FULL_RAGAS = "full_ragas"
    LIGHTWEIGHT = "lightweight" 
    HYBRID = "hybrid"
    MONITORING_ONLY = "monitoring_only"

@dataclass
class RAGEvaluationResult:
    """Container for RAG evaluation results."""
    query: str
    retrieved_contexts: List[str]
    generated_answer: str
    ground_truth: Optional[str]
    
    # RAGAS metrics (if available)
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    context_precision_score: Optional[float] = None
    context_recall_score: Optional[float] = None
    answer_similarity_score: Optional[float] = None
    answer_correctness_score: Optional[float] = None
    
    # Lightweight metrics
    retrieval_precision: Optional[float] = None
    retrieval_recall: Optional[float] = None
    answer_bert_score: Optional[float] = None
    answer_bleu_score: Optional[float] = None
    context_relevance_score: Optional[float] = None
    
    # Metadata
    evaluation_mode: str = EvaluationMode.LIGHTWEIGHT.value
    evaluation_timestamp: datetime = None
    response_time_ms: Optional[float] = None
    token_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/monitoring."""
        return {
            "query": self.query,
            "retrieved_contexts_count": len(self.retrieved_contexts),
            "generated_answer_length": len(self.generated_answer),
            "ground_truth_available": self.ground_truth is not None,
            "faithfulness_score": self.faithfulness_score,
            "answer_relevancy_score": self.answer_relevancy_score,
            "context_precision_score": self.context_precision_score,
            "context_recall_score": self.context_recall_score,
            "answer_similarity_score": self.answer_similarity_score,
            "answer_correctness_score": self.answer_correctness_score,
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "answer_bert_score": self.answer_bert_score,
            "answer_bleu_score": self.answer_bleu_score,
            "context_relevance_score": self.context_relevance_score,
            "evaluation_mode": self.evaluation_mode,
            "evaluation_timestamp": self.evaluation_timestamp.isoformat() if self.evaluation_timestamp else None,
            "response_time_ms": self.response_time_ms,
            "token_count": self.token_count
        }

class RAGEvaluationService:
    """Service for evaluating RAG system quality using RAGAS and lightweight alternatives."""
    
    def __init__(self, evaluation_mode: EvaluationMode = EvaluationMode.HYBRID):
        self.logger = logging.getLogger("RAGEvaluationService")
        self.evaluation_mode = evaluation_mode
        self.embedding_model = None
        
        # Initialize components based on available frameworks
        self._initialize_evaluation_components()
        
        # Evaluation history for trending analysis
        self.evaluation_history: List[RAGEvaluationResult] = []
        self.max_history_size = 1000
        
    def _initialize_evaluation_components(self):
        """Initialize evaluation components based on availability."""
        try:
            # Initialize embedding model for lightweight evaluation
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.logger.info("✅ Sentence transformer initialized for evaluation")
            
            # Download NLTK data if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
                
            # Log available evaluation capabilities
            capabilities = ["lightweight_metrics", "embedding_similarity"]
            if RAGAS_AVAILABLE:
                capabilities.append("ragas_framework")
            if BERT_SCORE_AVAILABLE:
                capabilities.append("bert_score")
                
            self.logger.info(f"📊 Evaluation capabilities: {capabilities}")
            
        except Exception as e:
            self.logger.error(f"Error initializing evaluation components: {e}")
    
    async def evaluate_rag_response(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RAGEvaluationResult:
        """
        Evaluate a RAG response using available evaluation methods.
        
        Args:
            query: User query
            retrieved_contexts: List of retrieved context strings
            generated_answer: Generated response from RAG system
            ground_truth: Optional ground truth answer for comparison
            metadata: Additional metadata (response time, tokens, etc.)
        """
        with monitor_agent_operation("rag_evaluation", "evaluate_response"):
            try:
                start_time = datetime.now()
                
                # Create result container
                result = RAGEvaluationResult(
                    query=query,
                    retrieved_contexts=retrieved_contexts,
                    generated_answer=generated_answer,
                    ground_truth=ground_truth,
                    evaluation_mode=self.evaluation_mode.value,
                    evaluation_timestamp=start_time
                )
                
                # Add metadata if provided
                if metadata:
                    result.response_time_ms = metadata.get('response_time_ms')
                    result.token_count = metadata.get('token_count')
                
                # Run evaluation based on mode and availability
                if self.evaluation_mode == EvaluationMode.FULL_RAGAS and RAGAS_AVAILABLE:
                    await self._run_ragas_evaluation(result)
                elif self.evaluation_mode == EvaluationMode.LIGHTWEIGHT:
                    await self._run_lightweight_evaluation(result)
                elif self.evaluation_mode == EvaluationMode.HYBRID:
                    # Try RAGAS first, fallback to lightweight
                    if RAGAS_AVAILABLE:
                        await self._run_ragas_evaluation(result)
                    await self._run_lightweight_evaluation(result)
                elif self.evaluation_mode == EvaluationMode.MONITORING_ONLY:
                    # Only collect basic metrics for monitoring
                    await self._run_monitoring_evaluation(result)
                
                # Store in history
                self._add_to_history(result)
                
                # Record evaluation metrics for monitoring
                self._record_evaluation_metrics(result)
                
                evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
                self.logger.info(f"RAG evaluation completed in {evaluation_time:.2f}ms")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error in RAG evaluation: {e}")
                record_article_processing("evaluation", "error")
                
                # Return basic result with error info
                return RAGEvaluationResult(
                    query=query,
                    retrieved_contexts=retrieved_contexts,
                    generated_answer=generated_answer,
                    ground_truth=ground_truth,
                    evaluation_mode="error"
                )
    
    async def _run_ragas_evaluation(self, result: RAGEvaluationResult):
        """Run RAGAS framework evaluation."""
        try:
            # Prepare dataset for RAGAS
            eval_data = {
                "question": [result.query],
                "contexts": [result.retrieved_contexts],
                "answer": [result.generated_answer]
            }
            
            # Add ground truth if available
            if result.ground_truth:
                eval_data["ground_truth"] = [result.ground_truth]
            
            dataset = Dataset.from_dict(eval_data)
            
            # Run RAGAS evaluation
            metrics_to_evaluate = [
                faithfulness,
                answer_relevancy,
                context_precision
            ]
            
            # Add metrics that require ground truth
            if result.ground_truth:
                metrics_to_evaluate.extend([
                    context_recall,
                    answer_semantic_similarity,
                    answer_correctness
                ])
            
            # Run evaluation (this might take a few seconds)
            evaluation_result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: evaluate(dataset, metrics=metrics_to_evaluate)
            )
            
            # Extract scores
            result.faithfulness_score = evaluation_result['faithfulness'][0] if 'faithfulness' in evaluation_result else None
            result.answer_relevancy_score = evaluation_result['answer_relevancy'][0] if 'answer_relevancy' in evaluation_result else None
            result.context_precision_score = evaluation_result['context_precision'][0] if 'context_precision' in evaluation_result else None
            
            if result.ground_truth:
                result.context_recall_score = evaluation_result['context_recall'][0] if 'context_recall' in evaluation_result else None
                result.answer_similarity_score = evaluation_result['answer_semantic_similarity'][0] if 'answer_semantic_similarity' in evaluation_result else None
                result.answer_correctness_score = evaluation_result['answer_correctness'][0] if 'answer_correctness' in evaluation_result else None
            
            self.logger.info(f"RAGAS evaluation completed - Faithfulness: {result.faithfulness_score:.3f}")
            
        except Exception as e:
            self.logger.warning(f"RAGAS evaluation failed: {e}")
    
    async def _run_lightweight_evaluation(self, result: RAGEvaluationResult):
        """Run lightweight evaluation using simpler metrics."""
        try:
            # 1. Context relevance using embedding similarity
            if self.embedding_model and result.retrieved_contexts:
                query_embedding = self.embedding_model.encode([result.query])
                context_embeddings = self.embedding_model.encode(result.retrieved_contexts)
                
                # Calculate average similarity of contexts to query
                similarities = cosine_similarity(query_embedding, context_embeddings)[0]
                result.context_relevance_score = float(np.mean(similarities))
            
            # 2. Answer-query relevance
            if self.embedding_model:
                query_embedding = self.embedding_model.encode([result.query])
                answer_embedding = self.embedding_model.encode([result.generated_answer])
                answer_query_similarity = cosine_similarity(query_embedding, answer_embedding)[0][0]
                
                # Use this as a proxy for answer relevancy
                if result.answer_relevancy_score is None:
                    result.answer_relevancy_score = float(answer_query_similarity)
            
            # 3. BERT Score if available and ground truth exists
            if BERT_SCORE_AVAILABLE and result.ground_truth:
                P, R, F1 = bert_score([result.generated_answer], [result.ground_truth], lang="en")
                result.answer_bert_score = float(F1[0])
            
            # 4. Simple BLEU score approximation
            if result.ground_truth:
                result.answer_bleu_score = self._calculate_simple_bleu(
                    result.generated_answer, 
                    result.ground_truth
                )
            
            # 5. Retrieval metrics
            if result.retrieved_contexts:
                # Simple precision based on non-empty contexts
                non_empty_contexts = [c for c in result.retrieved_contexts if c.strip()]
                result.retrieval_precision = len(non_empty_contexts) / len(result.retrieved_contexts)
                
                # Simple recall approximation (assume we got relevant contexts if similarity > threshold)
                if hasattr(result, 'context_relevance_score') and result.context_relevance_score:
                    result.retrieval_recall = min(1.0, result.context_relevance_score * 1.2)  # Rough approximation
            
            self.logger.info(f"Lightweight evaluation completed - Context relevance: {result.context_relevance_score:.3f}")
            
        except Exception as e:
            self.logger.warning(f"Lightweight evaluation failed: {e}")
    
    async def _run_monitoring_evaluation(self, result: RAGEvaluationResult):
        """Run basic evaluation for monitoring purposes only."""
        try:
            # Just collect basic statistics
            result.retrieval_precision = 1.0 if result.retrieved_contexts else 0.0
            result.context_relevance_score = 0.8  # Default assumption for monitoring
            
            if result.generated_answer:
                # Simple answer quality heuristic
                answer_length = len(result.generated_answer.split())
                result.answer_relevancy_score = min(1.0, answer_length / 50.0)  # Normalize by expected length
            
        except Exception as e:
            self.logger.warning(f"Monitoring evaluation failed: {e}")
    
    def _calculate_simple_bleu(self, candidate: str, reference: str) -> float:
        """Calculate a simple BLEU-like score."""
        try:
            candidate_tokens = set(candidate.lower().split())
            reference_tokens = set(reference.lower().split())
            
            if not reference_tokens:
                return 0.0
            
            overlap = len(candidate_tokens.intersection(reference_tokens))
            return overlap / len(reference_tokens)
            
        except Exception:
            return 0.0
    
    def _add_to_history(self, result: RAGEvaluationResult):
        """Add evaluation result to history."""
        self.evaluation_history.append(result)
        
        # Trim history if too large
        if len(self.evaluation_history) > self.max_history_size:
            self.evaluation_history = self.evaluation_history[-self.max_history_size:]
    
    def _record_evaluation_metrics(self, result: RAGEvaluationResult):
        """Record evaluation metrics for monitoring."""
        try:
            # Record basic evaluation completion
            record_article_processing("evaluation", "success")
            
            # Record specific metrics as custom monitoring metrics
            # (These could be added to monitoring.py as custom gauges)
            
            agent_operations_total.labels(
                agent_name="rag_evaluation",
                operation_type="quality_assessment",
                status="success"
            ).inc()
            
        except Exception as e:
            self.logger.warning(f"Failed to record evaluation metrics: {e}")
    
    def get_evaluation_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get evaluation trends over specified time period."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_evaluations = [
                r for r in self.evaluation_history 
                if r.evaluation_timestamp and r.evaluation_timestamp > cutoff_time
            ]
            
            if not recent_evaluations:
                return {"message": "No recent evaluations", "count": 0}
            
            # Calculate trends
            trends = {
                "total_evaluations": len(recent_evaluations),
                "average_scores": {},
                "score_trends": {},
                "evaluation_modes": {}
            }
            
            # Calculate averages
            score_fields = [
                'faithfulness_score', 'answer_relevancy_score', 'context_precision_score',
                'context_recall_score', 'answer_similarity_score', 'context_relevance_score',
                'retrieval_precision', 'answer_bert_score'
            ]
            
            for field in score_fields:
                scores = [getattr(r, field) for r in recent_evaluations if getattr(r, field) is not None]
                if scores:
                    trends["average_scores"][field] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores),
                        "min": np.min(scores),
                        "max": np.max(scores),
                        "count": len(scores)
                    }
            
            # Count evaluation modes
            for result in recent_evaluations:
                mode = result.evaluation_mode
                trends["evaluation_modes"][mode] = trends["evaluation_modes"].get(mode, 0) + 1
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error calculating evaluation trends: {e}")
            return {"error": str(e)}
    
    async def create_evaluation_report(self, include_history: bool = True) -> Dict[str, Any]:
        """Create comprehensive evaluation report."""
        try:
            report = {
                "service_info": {
                    "evaluation_mode": self.evaluation_mode.value,
                    "ragas_available": RAGAS_AVAILABLE,
                    "bert_score_available": BERT_SCORE_AVAILABLE,
                    "capabilities": []
                },
                "current_status": {
                    "total_evaluations": len(self.evaluation_history),
                    "last_evaluation": None
                }
            }
            
            # Add capabilities
            capabilities = ["lightweight_metrics"]
            if RAGAS_AVAILABLE:
                capabilities.append("ragas_framework")
            if BERT_SCORE_AVAILABLE:
                capabilities.append("bert_score")
            if self.embedding_model:
                capabilities.append("embedding_similarity")
            
            report["service_info"]["capabilities"] = capabilities
            
            # Add last evaluation info
            if self.evaluation_history:
                last_eval = self.evaluation_history[-1]
                report["current_status"]["last_evaluation"] = {
                    "timestamp": last_eval.evaluation_timestamp.isoformat() if last_eval.evaluation_timestamp else None,
                    "query_length": len(last_eval.query),
                    "contexts_count": len(last_eval.retrieved_contexts),
                    "answer_length": len(last_eval.generated_answer),
                    "faithfulness_score": last_eval.faithfulness_score,
                    "answer_relevancy_score": last_eval.answer_relevancy_score
                }
            
            # Add historical trends
            if include_history:
                report["trends_24h"] = self.get_evaluation_trends(24)
                report["trends_7d"] = self.get_evaluation_trends(24 * 7)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating evaluation report: {e}")
            return {"error": str(e)}

# Global service instance
rag_evaluation_service = RAGEvaluationService(
    evaluation_mode=EvaluationMode.HYBRID  # Use hybrid mode by default
)
