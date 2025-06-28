# Ollama Model Caching Strategy

This document outlines the caching strategy for Ollama models to avoid re-downloading large models on every Docker rebuild.

## Caching Mechanisms

### 1. Docker Volume Caching (`ollama_data`)
- **Location**: Docker-managed volume
- **Mount Point**: `/root/.ollama` (inside Ollama container)
- **Purpose**: Primary storage for Ollama models and configurations
- **Persistence**: Survives container restarts and rebuilds
- **Advantages**: 
  - Managed by Docker
  - Fast access from container
  - Automatically backed up with Docker volume backups

### 2. Host Filesystem Mount (`./models/ollama`)
- **Location**: `./models/ollama` (host filesystem)
- **Mount Point**: `/ollama_models` (inside Ollama container)
- **Purpose**: Additional cache and backup location
- **Persistence**: Permanent (survives Docker resets)
- **Advantages**:
  - Direct access from host
  - Can be backed up with project
  - Shared between different Docker environments

## Model Storage Locations

Models are cached in the following priority order:

1. **Primary**: `/root/.ollama/models` (Docker volume)
2. **Secondary**: `/ollama_models` (host mount)
3. **Fallback**: Download from Ollama registry

## Managing Models

### View Downloaded Models
```bash
# From host
docker exec gmail-search-ollama ollama list

# From inside container
ollama list
```

### Manual Model Management
```bash
# Pull a specific model
docker exec gmail-search-ollama ollama pull mistral:7b-instruct

# Remove a model (to free space)
docker exec gmail-search-ollama ollama rm mistral:7b-instruct

# Show model info
docker exec gmail-search-ollama ollama show mistral:7b-instruct
```

### Check Cache Status
```bash
# Check Docker volume usage
docker volume inspect gmail-article-search-agent_ollama_data

# Check host cache
ls -la ./models/ollama/

# Check available space
docker exec gmail-search-ollama df -h /root/.ollama
```

## Model Sizes

Common models and their approximate sizes:

| Model | Size | Context Window | Use Case |
|-------|------|----------------|----------|
| `mistral:7b-instruct` | ~4.1GB | 32k tokens | General purpose, RAG |
| `llama2:7b-chat` | ~3.8GB | 4k tokens | Chat, conversation |
| `codellama:7b` | ~3.8GB | 16k tokens | Code generation |
| `phi3:3.8b` | ~2.3GB | 128k tokens | Small, efficient |
| `gemma:2b` | ~1.4GB | 8k tokens | Very small, fast |

## Optimization Tips

### 1. Pre-download Models
Run the initialization script to download models before first use:
```bash
# Local environment
./init_ollama_docker.sh

# Docker environment
docker exec gmail-search-ollama python /app/setup_ollama_docker.py
```

### 2. Clean Up Unused Models
```bash
# List all models
docker exec gmail-search-ollama ollama list

# Remove unused models
docker exec gmail-search-ollama ollama rm <model_name>
```

### 3. Monitor Disk Usage
```bash
# Check Docker volume usage
docker system df -v

# Check Ollama specific usage
docker exec gmail-search-ollama du -sh /root/.ollama/
```

## Troubleshooting

### Cache Not Working
1. Check volume mounts:
   ```bash
   docker inspect gmail-search-ollama | grep -A 10 "Mounts"
   ```

2. Verify volume exists:
   ```bash
   docker volume ls | grep ollama
   ```

3. Check permissions:
   ```bash
   ls -la ./models/ollama/
   ```

### Models Not Found
1. Verify Ollama server is running:
   ```bash
   docker exec gmail-search-ollama ollama list
   ```

2. Check model download status:
   ```bash
   docker logs gmail-search-ollama
   ```

3. Manual download:
   ```bash
   docker exec gmail-search-ollama ollama pull mistral:7b-instruct
   ```

### Performance Issues
1. **CPU Usage**: Ollama uses CPU for inference on non-GPU systems
2. **Memory**: Ensure sufficient RAM (8GB+ recommended for 7B models)
3. **Storage**: SSD recommended for model loading speed

## Environment Variables

Configure caching behavior through environment variables:

```env
# Ollama model storage path
OLLAMA_MODELS=/root/.ollama/models

# Ollama host for API access
OLLAMA_HOST=http://ollama:11434

# Default model to use
OLLAMA_MODEL=mistral:7b-instruct

# Model download timeout (seconds)
OLLAMA_DOWNLOAD_TIMEOUT=1800
```

## Integration with Backend

The backend service automatically:

1. **Waits for Ollama**: Uses `depends_on` with health checks
2. **Connects to Models**: Via `OLLAMA_HOST` environment variable
3. **Handles Failures**: Graceful fallback if models unavailable
4. **Caches Results**: In-memory caching of frequent queries

## Best Practices

1. **Always use volumes** for model persistence
2. **Monitor disk space** - models can be large
3. **Use appropriate model sizes** for your hardware
4. **Pre-download models** during setup, not runtime
5. **Regular cleanup** of unused models
6. **Backup volumes** for production deployments

## Cache Reset

To completely reset the cache:

```bash
# Stop services
docker-compose down

# Remove volume (WARNING: This deletes all models)
docker volume rm gmail-article-search-agent_ollama_data

# Clear host cache
rm -rf ./models/ollama/*

# Restart services (will re-download models)
docker-compose up -d
```
