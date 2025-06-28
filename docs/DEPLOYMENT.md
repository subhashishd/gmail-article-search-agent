# Deployment Guide

## 🚀 **Production Deployment**

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB+ for article storage
- **CPU**: 4+ cores recommended
- **OS**: Linux (Ubuntu 20.04+ recommended)

### 1. Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Application Deployment
```bash
# Clone repository
git clone <your-repository-url>
cd gmail-article-search-agent

# Setup production environment
cp .env.example .env.production
# Edit .env.production with production values

# Deploy with production compose
docker-compose -f docker-compose.prod.yml up -d
```

### 3. SSL/TLS Setup
```bash
# Install Certbot
sudo apt install certbot

# Generate SSL certificates
sudo certbot certonly --standalone -d your-domain.com

# Configure nginx reverse proxy
# See nginx.conf example below
```

### 4. Monitoring Setup
```bash
# Deploy monitoring stack
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Configure alerting
# Edit monitoring/prometheus/alert.rules
```

## 🔒 **Security Configuration**

### Environment Variables
```bash
# Production .env file
DATABASE_URL=postgresql://secure_user:strong_password@db:5432/gmail_search
POSTGRES_PASSWORD=very_strong_password

# Enable security features
ENABLE_AUTH=true
JWT_SECRET=your-super-secret-jwt-key
CORS_ORIGINS=https://your-domain.com

# Monitoring security
GRAFANA_ADMIN_PASSWORD=secure_grafana_password
PROMETHEUS_AUTH_ENABLED=true
```

### Nginx Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Frontend
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Monitoring (restrict access)
    location /monitoring {
        auth_basic "Monitoring Area";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://localhost:3000;
    }
}
```

## 📊 **Monitoring & Alerting**

### Prometheus Alerting Rules
```yaml
# monitoring/prometheus/alerts.yml
groups:
- name: gmail-search-agent
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: High error rate detected
      
  - alert: AgentDown
    expr: up{job="gmail-search-backend"} == 0
    for: 2m
    annotations:
      summary: Backend service is down
```

### Grafana Alerts
- Configure notification channels (Slack, email)
- Set up dashboard alerts for key metrics
- Monitor resource usage and performance

## 🔄 **Backup & Recovery**

### Database Backup
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/gmail-search"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
docker-compose exec db pg_dump -U gmail_user gmail_search > "$BACKUP_DIR/backup_$DATE.sql"

# Compress and clean old backups
gzip "$BACKUP_DIR/backup_$DATE.sql"
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
```

### Application Data Backup
```bash
# Backup application data
docker run --rm -v gmail-search-data:/data -v $(pwd):/backup alpine tar czf /backup/data_backup.tar.gz -C /data .
```

## 📈 **Performance Optimization**

### Database Tuning
```sql
-- PostgreSQL optimization
-- In postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 16MB
maintenance_work_mem = 64MB
```

### Application Scaling
```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  backend:
    scale: 3
    environment:
      - WORKER_PROCESSES=4
  
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
      - "443:443"
```

## 🔧 **Maintenance**

### Regular Tasks
```bash
# Update containers
docker-compose pull
docker-compose up -d

# Clean up old images
docker system prune -f

# Monitor logs
docker-compose logs -f --tail=100

# Database maintenance
docker-compose exec db vacuumdb -U gmail_user -d gmail_search
```

### Health Checks
```bash
#!/bin/bash
# health_check.sh

# Check services
docker-compose ps | grep -q "Up" || echo "Services down!"

# Check API health
curl -f http://localhost:8000/health || echo "API unhealthy!"

# Check database
docker-compose exec db pg_isready -U gmail_user || echo "Database down!"
```

## 🚨 **Troubleshooting**

### Log Aggregation
```bash
# Centralized logging with ELK stack
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  elasticsearch:7.14.0

# Configure log shipping
# See logging/filebeat.yml
```

### Performance Monitoring
```bash
# Monitor system resources
htop
iotop
nethogs

# Docker monitoring
docker stats
docker system df
```

## 🔄 **CI/CD Pipeline**

### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to production
      run: |
        ssh production-server "cd /app && git pull && docker-compose up -d"
```

## 📋 **Checklist**

### Pre-deployment
- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] Database credentials secured
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] Alerting rules set up

### Post-deployment
- [ ] Health checks passing
- [ ] Monitoring dashboards working
- [ ] Log aggregation functional
- [ ] Backup process tested
- [ ] Performance baselines established
- [ ] Documentation updated

---

**Ready for production deployment! 🎉**
