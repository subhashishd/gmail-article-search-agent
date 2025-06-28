"""Monitoring and observability configuration for Gmail Article Search Agent."""

import os
import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

# Custom metrics for LLM operations
llm_requests_total = Counter(
    'llm_requests_total',
    'Total number of LLM requests',
    ['agent_name', 'model_name', 'operation_type']
)

llm_request_duration = Histogram(
    'llm_request_duration_seconds',
    'Duration of LLM requests in seconds',
    ['agent_name', 'model_name', 'operation_type']
)

llm_tokens_processed = Counter(
    'llm_tokens_processed_total',
    'Total number of tokens processed by LLM',
    ['agent_name', 'model_name', 'token_type']  # input_tokens, output_tokens
)

agent_operations_total = Counter(
    'agent_operations_total',
    'Total number of agent operations',
    ['agent_name', 'operation_type', 'status']
)

agent_operation_duration = Histogram(
    'agent_operation_duration_seconds',
    'Duration of agent operations in seconds',
    ['agent_name', 'operation_type']
)

articles_processed_total = Counter(
    'articles_processed_total',
    'Total number of articles processed',
    ['operation_type', 'status']  # fetch, index, search
)

database_operations_total = Counter(
    'database_operations_total',
    'Total number of database operations',
    ['operation_type', 'table_name', 'status']
)

database_operation_duration = Histogram(
    'database_operation_duration_seconds',
    'Duration of database operations in seconds',
    ['operation_type', 'table_name']
)

# Current state gauges
active_agents_gauge = Gauge('active_agents_current', 'Current number of active agents')
background_tasks_gauge = Gauge('background_tasks_current', 'Current number of background tasks')
articles_in_database_gauge = Gauge('articles_in_database_current', 'Current number of articles in database')

class MonitoringConfig:
    """Configuration for monitoring and observability."""
    
    def __init__(self):
        self.service_name = "gmail-article-search-agent"
        self.service_version = "1.0.0"
        self.environment = os.getenv("ENVIRONMENT", "local")
        
        # OpenTelemetry endpoints
        self.otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
        self.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")
        
        # Enable/disable features
        self.enable_tracing = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.enable_logging = os.getenv("ENABLE_LOGGING", "true").lower() == "true"

def setup_structured_logging():
    """Setup structured logging with structlog."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )
    
    return structlog.get_logger()

def setup_tracing(config: MonitoringConfig):
    """Setup OpenTelemetry tracing."""
    if not config.enable_tracing:
        return
    
    # Set up the tracer provider
    trace.set_tracer_provider(
        TracerProvider(
            resource=Resource.create({
                "service.name": config.service_name,
                "service.version": config.service_version,
                "deployment.environment": config.environment,
            })
        )
    )
    
    # Configure exporters
    try:
        # OTLP exporter (primary)
        otlp_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint, insecure=True)
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )
    except Exception as e:
        print(f"Failed to setup OTLP trace exporter: {e}")
    
    try:
        # Jaeger exporter (fallback)
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )
    except Exception as e:
        print(f"Failed to setup Jaeger trace exporter: {e}")

def setup_metrics(config: MonitoringConfig):
    """Setup OpenTelemetry metrics."""
    if not config.enable_metrics:
        return
    
    try:
        # OTLP metrics exporter
        otlp_metric_exporter = OTLPMetricExporter(endpoint=config.otlp_endpoint, insecure=True)
        metric_reader = PeriodicExportingMetricReader(otlp_metric_exporter, export_interval_millis=5000)
        
        metrics.set_meter_provider(
            MeterProvider(
                resource=Resource.create({
                    "service.name": config.service_name,
                    "service.version": config.service_version,
                    "deployment.environment": config.environment,
                }),
                metric_readers=[metric_reader]
            )
        )
    except Exception as e:
        print(f"Failed to setup metrics: {e}")

def setup_auto_instrumentation():
    """Setup automatic instrumentation for common libraries."""
    # Instrument HTTP libraries
    try:
        RequestsInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        print("✅ HTTP instrumentation enabled")
    except Exception as e:
        print(f"Failed to setup HTTP instrumentation: {e}")
    
    # Instrument database
    try:
        Psycopg2Instrumentor().instrument()
        print("✅ PostgreSQL instrumentation enabled")
    except Exception as e:
        print(f"Failed to setup PostgreSQL instrumentation: {e}")

def setup_fastapi_monitoring(app):
    """Setup FastAPI-specific monitoring."""
    # Prometheus metrics
    instrumentator = Instrumentator()
    instrumentator.instrument(app)
    instrumentator.expose(app, endpoint="/metrics")
    
    # OpenTelemetry FastAPI instrumentation
    try:
        FastAPIInstrumentor.instrument_app(app)
        print("✅ FastAPI instrumentation enabled")
    except Exception as e:
        print(f"Failed to setup FastAPI instrumentation: {e}")

def initialize_monitoring(app=None):
    """Initialize all monitoring components."""
    config = MonitoringConfig()
    
    # Setup structured logging
    logger = setup_structured_logging()
    logger.info("Setting up monitoring", service=config.service_name, version=config.service_version)
    
    # Setup tracing
    setup_tracing(config)
    logger.info("Tracing setup complete", enabled=config.enable_tracing)
    
    # Setup metrics
    setup_metrics(config)
    logger.info("Metrics setup complete", enabled=config.enable_metrics)
    
    # Setup auto-instrumentation
    setup_auto_instrumentation()
    
    # Note: FastAPI monitoring must be setup separately before app starts
    if app:
        logger.warning("FastAPI app provided but monitoring must be setup before app initialization")
    
    return logger

def setup_fastapi_monitoring_early(app):
    """Setup FastAPI monitoring early - must be called before app starts."""
    # Prometheus metrics
    instrumentator = Instrumentator()
    instrumentator.instrument(app)
    instrumentator.expose(app, endpoint="/metrics")
    
    # OpenTelemetry FastAPI instrumentation
    try:
        FastAPIInstrumentor.instrument_app(app)
        print("✅ FastAPI instrumentation enabled")
    except Exception as e:
        print(f"Failed to setup FastAPI instrumentation: {e}")

# Context managers for operation monitoring
@contextmanager
def monitor_llm_operation(agent_name: str, model_name: str, operation_type: str):
    """Monitor LLM operations with metrics and tracing."""
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span(f"llm_{operation_type}") as span:
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("llm.model_name", model_name)
        span.set_attribute("llm.operation_type", operation_type)
        
        start_time = time.time()
        status = "success"
        
        try:
            yield span
        except Exception as e:
            status = "error"
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            duration = time.time() - start_time
            
            # Record metrics
            llm_requests_total.labels(
                agent_name=agent_name,
                model_name=model_name,
                operation_type=operation_type
            ).inc()
            
            llm_request_duration.labels(
                agent_name=agent_name,
                model_name=model_name,
                operation_type=operation_type
            ).observe(duration)
            
            span.set_attribute("operation.duration", duration)
            span.set_attribute("operation.status", status)

@contextmanager
def monitor_agent_operation(agent_name: str, operation_type: str):
    """Monitor agent operations with metrics and tracing."""
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span(f"agent_{operation_type}") as span:
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("agent.operation_type", operation_type)
        
        start_time = time.time()
        status = "success"
        
        try:
            yield span
        except Exception as e:
            status = "error"
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            duration = time.time() - start_time
            
            # Record metrics
            agent_operations_total.labels(
                agent_name=agent_name,
                operation_type=operation_type,
                status=status
            ).inc()
            
            agent_operation_duration.labels(
                agent_name=agent_name,
                operation_type=operation_type
            ).observe(duration)
            
            span.set_attribute("operation.duration", duration)
            span.set_attribute("operation.status", status)

@contextmanager
def monitor_database_operation(operation_type: str, table_name: str):
    """Monitor database operations with metrics and tracing."""
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span(f"db_{operation_type}") as span:
        span.set_attribute("db.operation", operation_type)
        span.set_attribute("db.table", table_name)
        
        start_time = time.time()
        status = "success"
        
        try:
            yield span
        except Exception as e:
            status = "error"
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            duration = time.time() - start_time
            
            # Record metrics
            database_operations_total.labels(
                operation_type=operation_type,
                table_name=table_name,
                status=status
            ).inc()
            
            database_operation_duration.labels(
                operation_type=operation_type,
                table_name=table_name
            ).observe(duration)
            
            span.set_attribute("operation.duration", duration)
            span.set_attribute("operation.status", status)

# Utility functions for updating gauges
def update_system_metrics(agent_count: int, background_tasks: int, articles_count: int):
    """Update system-level gauge metrics."""
    active_agents_gauge.set(agent_count)
    background_tasks_gauge.set(background_tasks)
    articles_in_database_gauge.set(articles_count)

def record_article_processing(operation_type: str, status: str = "success"):
    """Record article processing metrics."""
    articles_processed_total.labels(
        operation_type=operation_type,
        status=status
    ).inc()

def record_llm_tokens(agent_name: str, model_name: str, input_tokens: int, output_tokens: int):
    """Record LLM token usage."""
    llm_tokens_processed.labels(
        agent_name=agent_name,
        model_name=model_name,
        token_type="input"
    ).inc(input_tokens)
    
    llm_tokens_processed.labels(
        agent_name=agent_name,
        model_name=model_name,
        token_type="output"
    ).inc(output_tokens)
