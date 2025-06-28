terraform {
  required_version = ">= 1.5.0"
  
  backend "gcs" {
    bucket = "your-terraform-state-bucket" # Replace with your bucket
    prefix = "gmail-search/dev"
  }
}

module "gmail_search" {
  source = "../../"
  
  project_id   = var.project_id
  region       = var.region
  environment  = "dev"
  image_tag    = var.image_tag
  db_password  = var.db_password
  
  # Development-specific overrides
  redis_memory_size_gb = 1
  cloud_sql_tier      = "db-f1-micro"
  max_scale_backend   = 2
  max_scale_frontend  = 1
  
  notification_channels = var.notification_channels
}
