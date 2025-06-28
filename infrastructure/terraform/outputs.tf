output "frontend_url" {
  description = "URL of the frontend Cloud Run service"
  value       = google_cloud_run_service.frontend.status[0].url
}

output "backend_url" {
  description = "URL of the backend Cloud Run service"
  value       = google_cloud_run_service.backend.status[0].url
}

output "database_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.main.connection_name
}

output "database_private_ip" {
  description = "Private IP address of the Cloud SQL instance"
  value       = google_sql_database_instance.main.private_ip_address
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.cache.host
}

output "redis_port" {
  description = "Redis instance port"
  value       = google_redis_instance.cache.port
}

output "vpc_network_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.vpc.name
}

output "service_account_email" {
  description = "Email of the Cloud Run service account"
  value       = google_service_account.cloud_run_service_account.email
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository for container images"
  value       = google_artifact_registry_repository.main.name
}

output "pubsub_topic_name" {
  description = "Name of the Pub/Sub topic for article events"
  value       = google_pubsub_topic.article_events.name
}

output "secret_manager_secrets" {
  description = "List of Secret Manager secret names"
  value = {
    gmail_credentials = google_secret_manager_secret.gmail_credentials.secret_id
    db_connection     = google_secret_manager_secret.db_connection.secret_id
  }
}
