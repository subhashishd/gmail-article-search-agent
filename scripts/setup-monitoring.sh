#!/bin/bash

# Setup Monitoring Stack for Gmail Article Search Agent
# This script sets up the complete observability stack

set -e

echo "🔍 Setting up Gmail Article Search Agent Monitoring Stack..."

# Create necessary directories
echo "📁 Creating monitoring directories..."
mkdir -p monitoring/prometheus/rules
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/loki
mkdir -p monitoring/promtail
mkdir -p monitoring/otel

# Check if Docker and Docker Compose are available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Start monitoring stack
echo "🚀 Starting monitoring stack..."
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for monitoring services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking monitoring service health..."

# Check Prometheus
if curl -f http://localhost:9090/-/ready &> /dev/null; then
    echo "✅ Prometheus is ready at http://localhost:9090"
else
    echo "⚠️  Prometheus may not be ready yet"
fi

# Check Grafana
if curl -f http://localhost:3000/api/health &> /dev/null; then
    echo "✅ Grafana is ready at http://localhost:3000 (admin/admin)"
else
    echo "⚠️  Grafana may not be ready yet"
fi

# Check Loki
if curl -f http://localhost:3100/ready &> /dev/null; then
    echo "✅ Loki is ready at http://localhost:3100"
else
    echo "⚠️  Loki may not be ready yet"
fi

# Check Jaeger
if curl -f http://localhost:16686/api/services &> /dev/null; then
    echo "✅ Jaeger UI is ready at http://localhost:16686"
else
    echo "⚠️  Jaeger may not be ready yet"
fi

echo ""
echo "🎉 Monitoring stack setup complete!"
echo ""
echo "📊 Access your monitoring tools:"
echo "  - Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Jaeger Tracing: http://localhost:16686"
echo "  - Loki Logs: Available via Grafana"
echo ""
echo "🔧 Next steps:"
echo "  1. Start your application with: docker-compose up -d"
echo "  2. Visit Grafana and import the LLM Agent dashboard"
echo "  3. Check metrics are flowing in Prometheus targets page"
echo "  4. Monitor traces in Jaeger"
echo ""
echo "📈 Key metrics to monitor:"
echo "  - LLM request rate and duration"
echo "  - Agent operation success/failure rates"
echo "  - Database operation performance"
echo "  - Article processing throughput"
echo "  - Token usage by LLM models"

# Go back to project root
cd ..

# Optionally start the main application
read -p "🤖 Start the main application now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting Gmail Article Search Agent..."
    docker-compose up -d
    
    echo "⏳ Waiting for application to be ready..."
    sleep 60
    
    # Check application health
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo "✅ Backend is ready at http://localhost:8000"
    else
        echo "⚠️  Backend may not be ready yet"
    fi
    
    if curl -f http://localhost:8501 &> /dev/null; then
        echo "✅ Frontend is ready at http://localhost:8501"
    else
        echo "⚠️  Frontend may not be ready yet"
    fi
    
    echo ""
    echo "🎯 Your Gmail Article Search Agent is now running with full monitoring!"
    echo "   Application: http://localhost:8501"
    echo "   API: http://localhost:8000"
    echo "   Monitoring: http://localhost:3000"
else
    echo "ℹ️  You can start the application later with: docker-compose up -d"
fi

echo ""
echo "✨ Happy monitoring! 🚀"
