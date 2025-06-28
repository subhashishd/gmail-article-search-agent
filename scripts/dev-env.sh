#!/bin/bash

# Gmail Article Search Agent - Development Environment Manager
# This script helps manage the development environment separately from production

set -e

COMPOSE_FILE="docker-compose.dev.yml"
ENV_FILE=".env.dev"
PROJECT_NAME="gmail-search-dev-env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} Gmail Article Search - Dev Environment${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    if [ ! -f "credentials/credentials.json" ]; then
        print_error "Gmail credentials not found at credentials/credentials.json"
        print_warning "Please ensure you have set up Gmail API credentials"
        exit 1
    fi
    
    print_status "Prerequisites check passed ✓"
}

build_env() {
    print_status "Building development environment..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME --env-file $ENV_FILE build --no-cache
    print_status "Build completed ✓"
}

start_env() {
    print_status "Starting development environment..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME --env-file $ENV_FILE up -d
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    print_status "Development environment started ✓"
    print_services_info
}

stop_env() {
    print_status "Stopping development environment..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down
    print_status "Environment stopped ✓"
}

restart_env() {
    print_status "Restarting development environment..."
    stop_env
    start_env
}

status_env() {
    print_status "Development environment status:"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
}

logs_env() {
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f
}

test_services() {
    print_status "Testing services..."
    
    # Test database
    print_status "Testing database connection..."
    if docker exec gmail-search-db-dev pg_isready -U postgres > /dev/null 2>&1; then
        print_status "Database: ✓ Ready"
    else
        print_error "Database: ✗ Not ready"
        return 1
    fi
    
    # Test backend
    print_status "Testing backend service..."
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        print_status "Backend: ✓ Ready"
    else
        print_error "Backend: ✗ Not ready"
        return 1
    fi
    
    # Test frontend
    print_status "Testing frontend service..."
    if curl -f http://localhost:8502/_stcore/health > /dev/null 2>&1; then
        print_status "Frontend: ✓ Ready"
    else
        print_error "Frontend: ✗ Not ready"
        return 1
    fi
    
    print_status "All services are healthy ✓"
}

test_gmail_connection() {
    print_status "Testing Gmail API connection..."
    
    response=$(curl -s -X POST http://localhost:8001/test-gmail \
        -H "Content-Type: application/json" \
        -d '{}')
    
    if echo "$response" | grep -q '"success":true'; then
        print_status "Gmail API: ✓ Connection successful"
    else
        print_error "Gmail API: ✗ Connection failed"
        echo "Response: $response"
        return 1
    fi
}

print_services_info() {
    echo ""
    echo -e "${GREEN}🚀 Development Environment Ready!${NC}"
    echo ""
    echo -e "${BLUE}Service URLs:${NC}"
    echo -e "  📱 Frontend (Streamlit): ${YELLOW}http://localhost:8502${NC}"
    echo -e "  🔧 Backend API: ${YELLOW}http://localhost:8001${NC}"
    echo -e "  📚 API Documentation: ${YELLOW}http://localhost:8001/docs${NC}"
    echo -e "  🗄️  Database: ${YELLOW}localhost:5433${NC}"
    echo ""
    echo -e "${BLUE}Container Names:${NC}"
    echo -e "  📦 Database: gmail-search-db-dev"
    echo -e "  📦 Backend: gmail-search-backend-dev"
    echo -e "  📦 Frontend: gmail-search-frontend-dev"
    echo ""
}

cleanup_env() {
    print_status "Cleaning up development environment..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v --remove-orphans
    docker system prune -f
    print_status "Cleanup completed ✓"
}

show_help() {
    echo "Gmail Article Search Agent - Development Environment Manager"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     Build the development environment"
    echo "  start     Start the development environment"
    echo "  stop      Stop the development environment"
    echo "  restart   Restart the development environment"
    echo "  status    Show environment status"
    echo "  logs      Show and follow logs"
    echo "  test      Test all services"
    echo "  test-gmail Test Gmail API connection"
    echo "  cleanup   Clean up environment and volumes"
    echo "  help      Show this help message"
    echo ""
}

# Main script logic
case "${1:-help}" in
    build)
        print_header
        check_prerequisites
        build_env
        ;;
    start)
        print_header
        check_prerequisites
        start_env
        ;;
    stop)
        print_header
        stop_env
        ;;
    restart)
        print_header
        restart_env
        ;;
    status)
        print_header
        status_env
        ;;
    logs)
        print_header
        logs_env
        ;;
    test)
        print_header
        test_services
        ;;
    test-gmail)
        print_header
        test_gmail_connection
        ;;
    cleanup)
        print_header
        cleanup_env
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
