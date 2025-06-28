terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.84"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.84"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "cloudsql.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "pubsub.googleapis.com",
    "redis.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com"
  ])
  
  service = each.value
  disable_dependent_services = false
}

# Artifact Registry Repository
resource "google_artifact_registry_repository" "main" {
  provider      = google-beta
  location      = var.region
  repository_id = "gmail-article-search"
  description   = "Container registry for Gmail Article Search Agent"
  format        = "DOCKER"
  
  depends_on = [google_project_service.required_apis]
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "${var.environment}-gmail-search-vpc"
  auto_create_subnetworks = false
  
  depends_on = [google_project_service.required_apis]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "${var.environment}-gmail-search-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
  
  private_ip_google_access = true
}

# Cloud SQL Instance
resource "google_sql_database_instance" "main" {
  name             = "${var.environment}-gmail-search-db"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier              = var.environment == "prod" ? "db-custom-2-8192" : "db-f1-micro"
    disk_size         = var.environment == "prod" ? 100 : 20
    disk_type         = "PD_SSD"
    availability_type = var.environment == "prod" ? "REGIONAL" : "ZONAL"
    
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = var.environment == "prod"
      backup_retention_settings {
        retained_backups = var.environment == "prod" ? 7 : 3
      }
    }
    
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = google_compute_network.vpc.id
      enable_private_path_for_google_cloud_services = true
    }
    
    database_flags {
      name  = "shared_preload_libraries"
      value = "vector"
    }
  }
  
  deletion_protection = var.environment == "prod"
  
  depends_on = [
    google_project_service.required_apis,
    google_service_networking_connection.private_vpc_connection
  ]
}

# Private VPC Connection for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.environment}-private-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# Cloud SQL Database
resource "google_sql_database" "database" {
  name     = "gmail_search"
  instance = google_sql_database_instance.main.name
}

# Cloud SQL User
resource "google_sql_user" "user" {
  name     = "gmail_search_user"
  instance = google_sql_database_instance.main.name
  password = var.db_password
}

# Redis (Memorystore)
resource "google_redis_instance" "cache" {
  name               = "${var.environment}-gmail-search-redis"
  memory_size_gb     = var.environment == "prod" ? 4 : 1
  region             = var.region
  redis_version      = "REDIS_7_0"
  tier               = var.environment == "prod" ? "STANDARD_HA" : "BASIC"
  
  authorized_network = google_compute_network.vpc.id
  
  depends_on = [google_project_service.required_apis]
}

# Pub/Sub Topics
resource "google_pubsub_topic" "article_events" {
  name = "${var.environment}-article-events"
  
  depends_on = [google_project_service.required_apis]
}

resource "google_pubsub_subscription" "content_agent_subscription" {
  name  = "${var.environment}-content-agent-subscription"
  topic = google_pubsub_topic.article_events.name
  
  ack_deadline_seconds = 300
  
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "300s"
  }
}

# Secret Manager Secrets
resource "google_secret_manager_secret" "gmail_credentials" {
  secret_id = "${var.environment}-gmail-credentials"
  
  replication {
    automatic = true
  }
  
  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret" "db_connection" {
  secret_id = "${var.environment}-db-connection"
  
  replication {
    automatic = true
  }
  
  depends_on = [google_project_service.required_apis]
}

# Service Account for Cloud Run
resource "google_service_account" "cloud_run_service_account" {
  account_id   = "${var.environment}-gmail-search-sa"
  display_name = "Gmail Search Agent Service Account"
}

# IAM Bindings for Service Account
resource "google_project_iam_member" "cloud_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.cloud_run_service_account.email}"
}

resource "google_project_iam_member" "secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.cloud_run_service_account.email}"
}

resource "google_project_iam_member" "pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.cloud_run_service_account.email}"
}

resource "google_project_iam_member" "pubsub_subscriber" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.cloud_run_service_account.email}"
}

resource "google_project_iam_member" "monitoring_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.cloud_run_service_account.email}"
}

# Cloud Run Services
resource "google_cloud_run_service" "backend" {
  name     = "${var.environment}-gmail-search-backend"
  location = var.region
  
  template {
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale"        = var.environment == "prod" ? "10" : "3"
        "run.googleapis.com/cloudsql-instances"   = google_sql_database_instance.main.connection_name
        "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.connector.name
      }
    }
    
    spec {
      service_account_name = google_service_account.cloud_run_service_account.email
      
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/gmail-article-search/backend:${var.image_tag}"
        
        ports {
          container_port = 8000
        }
        
        env {
          name  = "ENVIRONMENT"
          value = var.environment
        }
        
        env {
          name  = "DB_HOST"
          value = "/cloudsql/${google_sql_database_instance.main.connection_name}"
        }
        
        env {
          name  = "DB_NAME"
          value = google_sql_database.database.name
        }
        
        env {
          name  = "DB_USER"
          value = google_sql_user.user.name
        }
        
        env {
          name = "DB_PASS"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.db_connection.secret_id
              key  = "latest"
            }
          }
        }
        
        env {
          name  = "REDIS_HOST"
          value = google_redis_instance.cache.host
        }
        
        env {
          name  = "REDIS_PORT"
          value = "6379"
        }
        
        env {
          name  = "PUBSUB_TOPIC"
          value = google_pubsub_topic.article_events.name
        }
        
        resources {
          limits = {
            cpu    = var.environment == "prod" ? "2" : "1"
            memory = var.environment == "prod" ? "4Gi" : "2Gi"
          }
          requests = {
            cpu    = var.environment == "prod" ? "1" : "0.5"
            memory = var.environment == "prod" ? "2Gi" : "1Gi"
          }
        }
      }
    }
  }
  
  traffic {
    percent         = 100
    latest_revision = true
  }
  
  depends_on = [google_project_service.required_apis]
}

resource "google_cloud_run_service" "frontend" {
  name     = "${var.environment}-gmail-search-frontend"
  location = var.region
  
  template {
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = var.environment == "prod" ? "5" : "2"
      }
    }
    
    spec {
      service_account_name = google_service_account.cloud_run_service_account.email
      
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/gmail-article-search/frontend:${var.image_tag}"
        
        ports {
          container_port = 8501
        }
        
        env {
          name  = "BACKEND_URL"
          value = google_cloud_run_service.backend.status[0].url
        }
        
        resources {
          limits = {
            cpu    = "1"
            memory = "1Gi"
          }
          requests = {
            cpu    = "0.5"
            memory = "512Mi"
          }
        }
      }
    }
  }
  
  traffic {
    percent         = 100
    latest_revision = true
  }
  
  depends_on = [google_project_service.required_apis]
}

# VPC Connector for Cloud Run
resource "google_vpc_access_connector" "connector" {
  name          = "${var.environment}-vpc-connector"
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.vpc.name
  region        = var.region
  
  depends_on = [google_project_service.required_apis]
}

# IAM for public access (configure as needed)
resource "google_cloud_run_service_iam_member" "frontend_public" {
  location = google_cloud_run_service.frontend.location
  project  = google_cloud_run_service.frontend.project
  service  = google_cloud_run_service.frontend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Cloud Monitoring Alert Policies
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "${var.environment} High Error Rate"
  
  conditions {
    display_name = "High error rate"
    
    condition_threshold {
      filter         = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\""
      duration       = "300s"
      comparison     = "COMPARISON_GT"
      threshold_value = 10
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }
  
  notification_channels = var.notification_channels
  
  depends_on = [google_project_service.required_apis]
}
