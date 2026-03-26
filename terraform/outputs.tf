output "s3_bucket_name" {
  description = "S3 bucket name"
  value       = module.s3_bucket.bucket_name
}

output "github_role_arn" {
  description = "ARN of the GitHub Actions IAM role"
  value       = module.iam.github_role_arn
}

output "sagemaker_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = module.iam.sagemaker_role_arn
}

output "mlflow_ec2_public_ip" {
  description = "Public IP of the MLflow EC2 instance"
  value       = module.ec2_mlflow.public_ip
}

output "mlflow_url" {
  description = "MLflow tracking server URL"
  value       = "http://${module.ec2_mlflow.public_ip}:5000"
}

# Copy these values directly into GitHub repository secrets
output "github_secrets" {
  description = "Values to set as GitHub repository secrets (Settings -> Secrets -> Actions)"
  value = {
    AWS_ROLE_ARN        = module.iam.github_role_arn
    AWS_REGION          = var.aws_region
    S3_BUCKET           = module.s3_bucket.bucket_name
    SAGEMAKER_ROLE_ARN  = module.iam.sagemaker_role_arn
    MLFLOW_TRACKING_URI = "http://${module.ec2_mlflow.public_ip}:5000"
  }
}
