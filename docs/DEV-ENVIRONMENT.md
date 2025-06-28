# Gmail Article Search Agent - Development Environment

## Overview
This development environment provides a separate, isolated instance of the Gmail Article Search Agent with the meaningful name `gmail-search-dev-env`. It's designed for testing and development purposes without interfering with any production setup.

## Environment Details

### Service Configuration
- **Project Name**: `gmail-search-dev-env`
- **Network**: `gmail-search-network-dev`
- **Volumes**: Separate dev volumes to avoid conflicts

### Port Mapping (to avoid conflicts with production)
- **Database**: `localhost:5433` → `container:5432`
- **Backend API**: `localhost:8001` → `container:8000`
- **Frontend**: `localhost:8502` → `container:8501`

### Container Names
- **Database**: `gmail-search-db-dev`
- **Backend**: `gmail-search-backend-dev`
- **Frontend**: `gmail-search-frontend-dev`

## Quick Start

### Using the Management Script
The development environment includes a convenient management script:

```bash
# Show help
./dev-env.sh help

# Build the environment
./dev-env.sh build

# Start all services
./dev-env.sh start

# Check status
./dev-env.sh status

# Test all services
./dev-env.sh test

# Test Gmail API connection
./dev-env.sh test-gmail

# View logs
./dev-env.sh logs

# Stop services
./dev-env.sh stop

# Clean up everything
./dev-env.sh cleanup
```

### Manual Docker Compose Commands
If you prefer using Docker Compose directly:

```bash
# Build services
docker-compose -f docker-compose.dev.yml -p gmail-search-dev-env --env-file .env.dev build

# Start services
docker-compose -f docker-compose.dev.yml -p gmail-search-dev-env --env-file .env.dev up -d

# Check status
docker-compose -f docker-compose.dev.yml -p gmail-search-dev-env ps

# View logs
docker-compose -f docker-compose.dev.yml -p gmail-search-dev-env logs -f

# Stop services
docker-compose -f docker-compose.dev.yml -p gmail-search-dev-env down
```

## Service URLs

Once running, you can access:

- **📱 Frontend (Streamlit)**: http://localhost:8502
- **🔧 Backend API**: http://localhost:8001
- **📚 API Documentation**: http://localhost:8001/docs
- **🗄️  Database**: localhost:5433

## Development Features

### Simplified Backend
The development environment uses a simplified backend (`main_simple.py`) that provides:
- ✅ Basic API endpoints for testing
- ✅ Database connectivity testing
- ✅ Gmail credentials validation
- ✅ Mock search functionality
- ✅ Health checks
- ✅ CORS enabled for frontend development

### Mock Data
The simplified backend provides mock search results for testing the frontend without requiring the full AI/ML stack.

### Environment Variables
Uses `.env.dev` with development-specific configuration:
- Separate database name: `gmail_article_search_dev`
- Development service URLs
- Isolated from production settings

## Files Structure

```
gmail-article-search-agent/
├── docker-compose.dev.yml          # Development Docker Compose
├── .env.dev                        # Development environment variables
├── requirements.simple.txt         # Simplified Python dependencies
├── dev-env.sh                      # Development environment manager
├── backend/
│   ├── Dockerfile.dev              # Development backend Dockerfile
│   └── main_simple.py              # Simplified backend for testing
├── frontend/
│   └── Dockerfile.dev              # Development frontend Dockerfile
└── DEV-ENVIRONMENT.md              # This documentation
```

## Testing

### Service Health
Test all services:
```bash
./dev-env.sh test
```

### API Testing
Test individual endpoints:
```bash
# Health check
curl http://localhost:8001/health

# Gmail connection test
curl -X POST http://localhost:8001/test-gmail

# Search test
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 3}'

# Stats
curl http://localhost:8001/stats
```

### Frontend Testing
1. Open http://localhost:8502 in your browser
2. Test the UI components
3. Verify backend connectivity

## Gmail API Setup Verification

The development environment verifies that your Gmail API credentials are properly configured:

1. **Credentials File**: Located at `./credentials/credentials.json`
2. **Mounted**: Available inside containers at `/app/credentials/credentials.json`
3. **Validation**: Use `./dev-env.sh test-gmail` to verify

## Database

### Connection Details
- **Host**: localhost:5433 (external) / db-dev:5432 (internal)
- **Database**: gmail_article_search_dev
- **User**: postgres
- **Password**: postgres

### pgvector Extension
The database includes the pgvector extension for vector similarity search, ready for when you implement the full AI/ML features.

## Development Workflow

1. **Setup**: `./dev-env.sh build`
2. **Start**: `./dev-env.sh start`
3. **Develop**: Make changes to your code
4. **Test**: `./dev-env.sh test`
5. **Debug**: `./dev-env.sh logs`
6. **Stop**: `./dev-env.sh stop`

## Troubleshooting

### Services Not Starting
```bash
# Check logs
./dev-env.sh logs

# Check status
./dev-env.sh status

# Restart specific service
docker-compose -f docker-compose.dev.yml -p gmail-search-dev-env restart backend-dev
```

### Port Conflicts
The development environment uses different ports (5433, 8001, 8502) to avoid conflicts with any production instance running on standard ports.

### Database Issues
```bash
# Check database logs
docker-compose -f docker-compose.dev.yml -p gmail-search-dev-env logs db-dev

# Connect to database directly
docker exec -it gmail-search-db-dev psql -U postgres -d gmail_article_search_dev
```

## Next Steps

Once the development environment is working:

1. **Implement Full Features**: Replace `main_simple.py` with the full implementation
2. **Add Dependencies**: Update `requirements.simple.txt` with complete dependencies
3. **Test Gmail Integration**: Implement real Gmail API calls
4. **Add Vector Search**: Implement semantic search with embeddings
5. **Frontend Enhancement**: Connect frontend to real backend features

## Notes

- This environment is isolated and won't affect any production setup
- Uses mock data for testing without requiring the full AI/ML stack
- Gmail credentials are mounted read-only for security
- All services include health checks for reliable deployment
- Simplified dependencies to avoid version conflicts during development
