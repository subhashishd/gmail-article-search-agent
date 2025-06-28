variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region to deploy resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "db_password" {
  description = "Password for the Cloud SQL database"
  type        = string
  sensitive   = true
}

variable "notification_channels" {
  description = "List of notification channels for alerts"
  type        = list(string)
  default     = []
}

variable "enable_apis" {
  description = "Enable required GCP APIs"
  type        = bool
  default     = true
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/24"
}

variable "redis_memory_size_gb" {
  description = "Memory size for Redis instance in GB"
  type        = number
  default     = 1
}

variable "cloud_sql_tier" {
  description = "Cloud SQL instance tier"
  type        = string
  default     = "db-f1-micro"
}

variable "max_scale_backend" {
  description = "Maximum scale for backend Cloud Run service"
  type        = number
  default     = 3
}

variable "max_scale_frontend" {
  description = "Maximum scale for frontend Cloud Run service" 
  type        = number
  default     = 2
}
