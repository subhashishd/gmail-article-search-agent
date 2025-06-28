terraform {
  required_version = ">= 1.5.0"
  
  backend "gcs" {
    bucket = "your-terraform-state-bucket" # Replace with your bucket
    prefix = "gmail-search/prod"
  }
}

module "gmail_search" {
  source = "../../"
  
  project_id   = var.project_id
  region       = var.region
  environment  = "prod"
  image_tag    = var.image_tag
  db_password  = var.db_password
  
  # Production-specific overrides
  redis_memory_size_gb = 4
  cloud_sql_tier      = "db-custom-2-8192"
  max_scale_backend   = 10
  max_scale_frontend  = 5
  
  notification_channels = var.notification_channels
}
