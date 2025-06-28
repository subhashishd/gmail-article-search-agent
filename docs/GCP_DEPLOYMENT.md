# GCP Deployment Guide

This guide explains how to deploy the Gmail Article Search Agent to Google Cloud Platform using Terraform and Cloud Run.

## 🏗️ Architecture Overview

The GCP deployment includes:

- **Cloud Run**: Containerized services for backend and frontend
- **Cloud SQL**: PostgreSQL database with pgvector extension
- **Memorystore**: Redis for event bus and caching
- **Pub/Sub**: Event messaging between agents
- **Secret Manager**: Secure credential storage
- **Artifact Registry**: Container image storage
- **VPC**: Private networking with security

## 📋 Prerequisites

### Required Tools
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
source ~/.bashrc

# Install Terraform
wget https://releases.hashicorp.com/terraform/1.5.0/terraform_1.5.0_linux_amd64.zip
unzip terraform_1.5.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Install Docker
sudo apt-get update
sudo apt-get install docker.io
sudo usermod -aG docker $USER
```

### GCP Setup
1. Create a new GCP project or select existing one
2. Enable billing for the project
3. Create a service account with required permissions
4. Download service account key JSON file

### Required IAM Permissions
Your service account needs these roles:
- Cloud SQL Admin
- Cloud Run Admin
- Secret Manager Admin
- Artifact Registry Admin
- Pub/Sub Admin
- Redis Admin
- Compute Network Admin
- Service Account User
- Project IAM Admin

## 🚀 Quick Deployment

### Using the Deploy Script
```bash
# Make script executable
chmod +x scripts/deploy-gcp.sh

# Deploy to development
./scripts/deploy-gcp.sh dev your-gcp-project-id

# Deploy to production
./scripts/deploy-gcp.sh prod your-gcp-project-id
```

### Manual Deployment

#### 1. Authenticate with GCP
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

#### 2. Enable Required APIs
```bash
gcloud services enable \
    cloudsql.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    pubsub.googleapis.com \
    redis.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    vpcaccess.googleapis.com
```

#### 3. Create Terraform State Bucket
```bash
gsutil mb gs://your-project-terraform-state
gsutil versioning set on gs://your-project-terraform-state
```

#### 4. Configure Terraform Backend
Update `infrastructure/terraform/environments/[env]/main.tf`:
```hcl
backend "gcs" {
  bucket = "your-project-terraform-state"
  prefix = "gmail-search/[env]"
}
```

#### 5. Deploy Infrastructure
```bash
cd infrastructure/terraform/environments/dev

terraform init
terraform plan -var="project_id=YOUR_PROJECT_ID" -var="db_password=SECURE_PASSWORD"
terraform apply
```

## 🔧 Configuration

### Environment Variables
The following environment variables are automatically configured:

**Backend Service:**
- `DB_HOST`: Cloud SQL connection
- `DB_NAME`: Database name
- `DB_USER`: Database user
- `DB_PASS`: Database password (from Secret Manager)
- `REDIS_HOST`: Redis instance host
- `REDIS_PORT`: Redis port
- `PUBSUB_TOPIC`: Pub/Sub topic name

**Frontend Service:**
- `BACKEND_URL`: Backend service URL

### Secret Manager Setup
After deployment, configure secrets:

```bash
# Upload Gmail credentials
gcloud secrets versions add dev-gmail-credentials \
    --data-file=credentials/client_secret.json

# Set database password
echo "your-secure-password" | \
    gcloud secrets versions add dev-db-connection --data-file=-
```

### Database Setup
Configure pgvector extension:
```bash
# Connect to Cloud SQL instance
gcloud sql connect dev-gmail-search-db --user=gmail_search_user

# In PostgreSQL shell
CREATE EXTENSION IF NOT EXISTS vector;
```

## 📊 Monitoring and Logging

### Cloud Monitoring
- **Metrics**: Automatically collected for Cloud Run, SQL, and Redis
- **Alerts**: Configured for high error rates and resource usage
- **Dashboards**: Custom dashboards for application metrics

### Cloud Logging
- **Structured Logs**: JSON format with correlation IDs
- **Log Aggregation**: Centralized logging for all services
- **Error Reporting**: Automatic error detection and alerting

### Access Logs
```bash
# View backend logs
gcloud logs tail "projects/YOUR_PROJECT/logs/run.googleapis.com%2Frequests" \
    --filter="resource.labels.service_name=dev-gmail-search-backend"

# View frontend logs  
gcloud logs tail "projects/YOUR_PROJECT/logs/run.googleapis.com%2Frequests" \
    --filter="resource.labels.service_name=dev-gmail-search-frontend"
```

## 🔒 Security Considerations

### Network Security
- **Private VPC**: All resources in private network
- **Private IP**: Cloud SQL only accessible via private IP
- **VPC Connector**: Secure connection between Cloud Run and VPC

### Access Control
- **Service Accounts**: Dedicated service accounts with minimal permissions
- **IAM Roles**: Principle of least privilege
- **Secret Manager**: Encrypted credential storage

### Data Protection
- **Encryption**: Data encrypted at rest and in transit
- **Backup**: Automated database backups
- **Audit Logs**: Complete audit trail of all operations

## 🚦 Scaling Configuration

### Development Environment
- **Cloud Run**: 1-2 instances
- **Cloud SQL**: db-f1-micro (1 vCPU, 0.6GB RAM)
- **Redis**: 1GB memory, Basic tier

### Production Environment
- **Cloud Run**: Auto-scaling 1-10 instances
- **Cloud SQL**: db-custom-2-8192 (2 vCPU, 8GB RAM)
- **Redis**: 4GB memory, Standard HA tier

### Auto-scaling Rules
```hcl
# Backend scaling
autoscaling.knative.dev/maxScale = "10"
autoscaling.knative.dev/minScale = "1"

# CPU and memory limits
resources {
  limits = {
    cpu    = "2"
    memory = "4Gi"
  }
}
```

## 💰 Cost Optimization

### Development Costs (Monthly)
- Cloud Run: ~$10-20
- Cloud SQL: ~$25-40
- Redis: ~$30-50
- **Total**: ~$65-110/month

### Production Costs (Monthly)
- Cloud Run: ~$50-200
- Cloud SQL: ~$200-400
- Redis: ~$150-300
- **Total**: ~$400-900/month

### Cost Reduction Tips
1. **Use preemptible instances** for development
2. **Enable auto-scaling** to scale to zero when idle
3. **Use committed use discounts** for production
4. **Monitor resource usage** with billing alerts

## 🔄 CI/CD Integration

### GitHub Actions
The repository includes GitHub Actions workflows for:
- Automated testing on PRs
- Building and pushing container images
- Deploying to development on `develop` branch
- Deploying to production on `main` branch

### Required Secrets
Configure these GitHub secrets:
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_SA_KEY`: Service account key JSON (base64 encoded)

## 🛠️ Troubleshooting

### Common Issues

#### 1. API Not Enabled Error
```bash
# Solution: Enable the required API
gcloud services enable [api-name].googleapis.com
```

#### 2. Insufficient Permissions
```bash
# Solution: Add required IAM roles
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
    --role="roles/ROLE_NAME"
```

#### 3. Cloud Run Deployment Fails
```bash
# Check Cloud Run logs
gcloud run services describe SERVICE_NAME --region=REGION
gcloud logs tail "projects/PROJECT_ID/logs/run.googleapis.com%2Frequests"
```

#### 4. Database Connection Issues
```bash
# Check Cloud SQL status
gcloud sql instances describe INSTANCE_NAME

# Verify VPC connector
gcloud compute networks vpc-access connectors list --region=REGION
```

### Health Checks
```bash
# Backend health check
curl https://BACKEND_URL/health

# Frontend health check  
curl https://FRONTEND_URL

# Database connectivity
gcloud sql connect INSTANCE_NAME --user=USER_NAME
```

## 📚 Additional Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud SQL for PostgreSQL](https://cloud.google.com/sql/docs/postgres)
- [Memorystore for Redis](https://cloud.google.com/memorystore/docs/redis)
- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google/latest)
- [GCP Architecture Center](https://cloud.google.com/architecture)

## 🆘 Support

For deployment issues:
1. Check the troubleshooting section above
2. Review Cloud Logging for error details
3. Verify all prerequisites are met
4. Create an issue in the repository with deployment logs
