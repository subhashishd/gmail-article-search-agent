#!/bin/bash

# Gmail Article Search Agent - GCP Deployment Script
# Usage: ./deploy-gcp.sh [dev|prod] [project-id]

set -e

ENVIRONMENT=${1:-dev}
PROJECT_ID=${2}
REGION=${REGION:-us-central1}

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: $0 [dev|prod] [project-id]"
    echo "Example: $0 dev my-gcp-project"
    exit 1
fi

echo "🚀 Deploying Gmail Article Search Agent to GCP"
echo "Environment: $ENVIRONMENT"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"

# Authenticate with GCP
echo "🔐 Authenticating with GCP..."
gcloud auth login
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable \
    cloudsql.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    pubsub.googleapis.com \
    redis.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    cloudresourcemanager.googleapis.com \
    iam.googleapis.com \
    vpcaccess.googleapis.com

# Create Terraform state bucket if it doesn't exist
STATE_BUCKET="${PROJECT_ID}-terraform-state"
echo "📦 Creating Terraform state bucket: $STATE_BUCKET"
gsutil mb -p $PROJECT_ID gs://$STATE_BUCKET 2>/dev/null || echo "Bucket already exists"
gsutil versioning set on gs://$STATE_BUCKET

# Update Terraform backend configuration
TERRAFORM_DIR="infrastructure/terraform/environments/$ENVIRONMENT"
sed -i.bak "s/your-terraform-state-bucket/$STATE_BUCKET/g" $TERRAFORM_DIR/main.tf

# Generate database password
DB_PASSWORD=$(openssl rand -base64 32)
echo "🔑 Generated database password"

# Initialize and apply Terraform
echo "🏗️ Initializing Terraform..."
cd $TERRAFORM_DIR
terraform init

echo "📋 Planning Terraform deployment..."
terraform plan \
    -var="project_id=$PROJECT_ID" \
    -var="region=$REGION" \
    -var="db_password=$DB_PASSWORD" \
    -out=tfplan

echo "🚀 Applying Terraform configuration..."
terraform apply tfplan

# Get outputs
BACKEND_URL=$(terraform output -raw backend_url)
FRONTEND_URL=$(terraform output -raw frontend_url)
DB_CONNECTION=$(terraform output -raw database_connection_name)

echo "✅ Deployment completed!"
echo ""
echo "📋 Deployment Summary:"
echo "Environment: $ENVIRONMENT"
echo "Frontend URL: $FRONTEND_URL"
echo "Backend URL: $BACKEND_URL"
echo "Database: $DB_CONNECTION"
echo ""
echo "🔐 Next steps:"
echo "1. Upload Gmail credentials to Secret Manager:"
echo "   gcloud secrets versions add ${ENVIRONMENT}-gmail-credentials --data-file=credentials/client_secret.json"
echo ""
echo "2. Set database password in Secret Manager:"
echo "   echo '$DB_PASSWORD' | gcloud secrets versions add ${ENVIRONMENT}-db-connection --data-file=-"
echo ""
echo "3. Configure pgvector extension in Cloud SQL:"
echo "   gcloud sql databases patch gmail_search --instance=${ENVIRONMENT}-gmail-search-db --project=$PROJECT_ID"
echo ""
echo "4. Access the application:"
echo "   Frontend: $FRONTEND_URL"
echo "   Backend API: $BACKEND_URL"

cd ../../../..
