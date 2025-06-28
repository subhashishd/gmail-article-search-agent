# Gmail Article Search Agent

A sophisticated multi-agent AI system for intelligent Gmail article discovery, processing, and search using event-driven architecture and advanced RAG techniques.

## 🎯 What is Agentic AI and Multi-Agent Architecture?

### Agentic AI
**Agentic AI** refers to AI systems that can act autonomously, make decisions, and interact with their environment to achieve specific goals. Unlike traditional AI that simply responds to inputs, agentic systems:
- Make independent decisions based on objectives
- Adapt to changing conditions
- Coordinate with other agents
- Learn from interactions and feedback

### Multi-Agent Architecture
**Multi-agent systems** consist of multiple autonomous agents that work together to solve complex problems. Each agent:
- Has specialized responsibilities and capabilities
- Communicates with other agents through well-defined protocols
- Can operate independently while contributing to collective goals
- Enables parallel processing and distributed problem-solving

## 🏗️ How This System Embodies Multi-Agent Architecture

Our Gmail Article Search Agent implements a true multi-agent system with:

### **Agent Specialization**
- **Email Processor Agent**: Gmail integration, digest parsing, timeline management
- **Content Agent**: Parallel article fetching, rate limiting, content processing
- **Search Agent**: Hybrid RAG search, LLM contextualization, result caching

### **Event-Driven Communication**
- **Redis Event Bus**: Asynchronous pub/sub messaging between agents
- **Event Coordination**: Centralized event routing and monitoring
- **Loose Coupling**: Agents operate independently while staying synchronized

### **Autonomous Decision Making**
- **Rate Limiting**: Intelligent throttling based on API response patterns
- **Content Processing**: Automatic quality assessment and fallback strategies
- **Search Strategy**: Multi-strategy search with adaptive thresholds

## 🚀 System Functionality

### Core Capabilities

1. **Gmail Integration**
   - OAuth2 authentication with Gmail API
   - Chronological email processing with timestamp persistence
   - Medium digest email detection and parsing
   - Crash-safe resume capabilities

2. **Intelligent Content Processing**
   - Parallel article content fetching with rate limiting
   - Vector embedding generation using sentence transformers
   - LLM-powered article summarization and analysis
   - Duplicate detection and URL normalization

3. **Advanced Search**
   - Hybrid RAG (Retrieval-Augmented Generation) search
   - Vector similarity search with semantic understanding
   - Multi-strategy fallback (keyword, fuzzy, individual terms)
   - Real-time LLM contextualization and result ranking

4. **Robust Infrastructure**
   - Docker containerization with orchestrated services
   - Redis for event bus and caching
   - PostgreSQL with vector extensions (pgvector)
   - Comprehensive monitoring and observability

## 🔍 Search Strategy Deep Dive

### Multi-Strategy Search Pipeline

1. **Vector Search (Primary)**
   ```
   Query → Embedding → Vector Similarity → Ranked Results
   ```
   - Uses sentence-transformers for semantic understanding
   - Cosine similarity with adaptive thresholds
   - Pre-computed LLM summaries for fast response

2. **Individual Terms Search (Fallback 1)**
   ```
   "Claude Anthropic" → ["Claude", "Anthropic"] → Union Results
   ```
   - Breaks complex queries into individual terms
   - Handles cases where combined terms fail

3. **Keyword Search (Fallback 2)**
   ```
   PostgreSQL full-text search with BM25-like scoring
   ```
   - Traditional keyword matching
   - Multi-field search (title, content, summary)

4. **Fuzzy Search (Fallback 3)**
   ```
   PostgreSQL similarity functions (pg_trgm)
   ```
   - Handles typos and partial matches
   - Last resort for difficult queries

### LLM Enhancement Strategy

- **Background Processing**: Articles processed and summarized during indexing
- **Search-Time Enhancement**: Single LLM call for result set contextualization
- **Fallback Handling**: Graceful degradation when LLM services unavailable

## 📡 Redis Architecture Considerations

### Event Bus Design
```
Gmail Agent → Redis Pub/Sub → Content Agents
Content Agents → Redis Pub/Sub → Search Cache Updates
```

### Key Benefits
- **Scalability**: Easy to add more worker agents
- **Resilience**: Message persistence and replay capabilities
- **Monitoring**: Event tracking and flow observability

### Redis Usage Patterns
- **Event Bus**: Pub/Sub for agent communication
- **Rate Limiting**: Token bucket algorithm implementation
- **Search Caching**: Fast result retrieval with TTL
- **Session Management**: Worker coordination and status

## 📊 Observability and Monitoring

### Metrics Collection
- **Prometheus Integration**: Custom metrics for agent performance
- **Database Monitoring**: Query performance and connection health
- **LLM Performance**: Response times, token usage, error rates
- **Event Flow Tracking**: Message throughput and processing latency

### Key Metrics
```
- gmail_emails_processed_total
- articles_content_fetched_total  
- search_queries_executed_total
- llm_requests_total
- redis_events_published_total
```

### Logging Strategy
- **Structured Logging**: JSON format with correlation IDs
- **Multi-Level Logging**: Debug, Info, Warning, Error
- **Agent-Specific Logs**: Clear attribution and tracing

## 🔬 Evaluation and Inference Considerations

### RAG Evaluation Framework
- **Retrieval Quality**: Relevance scoring and precision/recall metrics
- **Generation Quality**: LLM output consistency and accuracy
- **End-to-End Performance**: Search relevance and user satisfaction

### Inference Optimization
- **Model Caching**: Warm model instances for reduced latency
- **Batch Processing**: Efficient LLM utilization for background tasks
- **Adaptive Strategies**: Quality vs. speed trade-offs based on load

## 🤖 AI Framework Compatibility

### LangChain Integration
Our multi-agent system is built on **LangChain 0.0.350**, providing:
- **Agent Framework**: Standard agent interfaces and message handling
- **Tool Integration**: Extensible capability system for agent tools
- **Memory Management**: Conversation history and context persistence
- **Chain Composition**: Complex workflow orchestration
- **Observability**: Built-in logging and monitoring integration

### LlamaIndex RAG Capabilities
Advanced RAG functionality powered by **LlamaIndex 0.9.0**:
- **Document Processing**: Intelligent text parsing and chunking
- **Index Management**: Efficient vector index creation and maintenance
- **Query Processing**: Sophisticated query understanding and routing
- **Response Synthesis**: Context-aware answer generation
- **Evaluation Framework**: Built-in RAG quality assessment

### Local LLM Support
Seamless integration with local LLM infrastructure:
- **Ollama Integration**: Direct API compatibility with Ollama service
- **Model Flexibility**: Support for Llama 3.2 1B and other models
- **Resource Optimization**: Efficient inference on limited hardware
- **Fallback Strategies**: Graceful degradation when LLM unavailable
- **Custom Prompting**: Specialized prompts for each agent's role

### Framework Interoperability
- **Standards Compliance**: Compatible with LangChain agent protocols
- **Extensible Architecture**: Easy to add new LangChain tools and chains
- **Migration Ready**: Seamless transition to cloud LLM providers
- **Hybrid Deployment**: Mix of local and cloud AI services

## 💻 Local Environment Adaptability

### Low-Resource Configuration
```yaml
# Minimal setup for development
services:
  - PostgreSQL (with reduced connections)
  - Redis (single instance)
  - Ollama (CPU-only LLM)
  - Single worker agents
```

### Resource Requirements
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores
- **Storage**: 10GB for models and data
- **Network**: Gmail API access required

### Development Mode
- Reduced worker concurrency
- Simplified LLM models (TinyLlama, etc.)
- Local file storage options
- Debug logging enabled

## ☁️ Cloud Migration Strategy

### GCP Deployment Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│   Cloud Run     │    │  Cloud SQL       │    │  Redis         │
│   (Agents)      │◄──►│  (PostgreSQL)    │    │  (Memorystore) │
└─────────────────┘    └──────────────────┘    └────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│   Pub/Sub       │    │  Cloud Storage   │    │  Secret        │
│   (Events)      │    │  (Models/Data)   │    │  Manager       │
└─────────────────┘    └──────────────────┘    └────────────────┘
```

### Migration Benefits
- **Auto-scaling**: Automatic agent scaling based on load
- **Managed Services**: Reduced operational overhead
- **High Availability**: Multi-zone deployment for resilience
- **Cost Optimization**: Pay-per-use pricing model

### Cloud Services Mapping
- **Compute**: Cloud Run for containerized agents
- **Database**: Cloud SQL for PostgreSQL with pgvector
- **Message Queue**: Cloud Pub/Sub for event bus
- **Caching**: Memorystore for Redis
- **Storage**: Cloud Storage for models and artifacts
- **Secrets**: Secret Manager for API keys
- **Monitoring**: Cloud Monitoring + Prometheus

## 🚀 Quick Start

### Prerequisites
```bash
# Required tools
- Docker & Docker Compose
- Python 3.9+
- Gmail API credentials
```

### Local Development
```bash
# 1. Clone and setup
git clone <repository>
cd gmail-article-search-agent

# 2. Environment setup
cp config/.env.example .env
# Edit .env with your credentials

# 3. Start services
cd config
docker-compose up -d

# 4. Run tests
cd ../tests
python run_tests.py
```

### Production Deployment
```bash
# Using provided Terraform scripts
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

## 📚 Documentation Structure

- [**Architecture**](docs/NEW_ARCHITECTURE.md) - Detailed system design
- [**Deployment**](docs/DEPLOYMENT.md) - Production deployment guide
- [**Development**](docs/DEV-ENVIRONMENT.md) - Local development setup
- [**Gmail Setup**](docs/setup-gmail-credentials.md) - API configuration
- [**Monitoring**](docs/OLLAMA_CACHING.md) - Performance optimization
- [**Troubleshooting**](docs/CHRONOLOGICAL_PROCESSING_FIX.md) - Common issues

## 🧪 Testing

### Test Suite Coverage
- **Unit Tests**: Individual agent functionality
- **Integration Tests**: Agent communication and coordination
- **System Tests**: End-to-end workflow validation
- **Performance Tests**: Load testing and benchmarking

### Running Tests
```bash
# All tests
cd tests && python run_tests.py

# Specific test categories
python test_new_architecture.py      # Core functionality
python test_docker_system.py         # Infrastructure
python comprehensive_system_test.py  # End-to-end
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for agentic AI framework and agent orchestration
- **LlamaIndex** for advanced RAG capabilities and document processing
- **Ollama** for local LLM inference (Llama 3.2 1B model)
- **Sentence Transformers** for embedding models
- **PostgreSQL** with pgvector for vector storage
- **Redis** for event coordination
- **Docker** for containerization

---

**Built with ❤️ for intelligent article discovery and search**
