variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name used in resource naming"
  type        = string
  default     = "mlops-wine"
}

variable "github_org" {
  description = "GitHub organization or username (e.g. romankovsv)"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name (e.g. mlops-sagemaker-wine)"
  type        = string
}

variable "github_oidc_provider_arn" {
  description = "ARN of the existing GitHub OIDC provider in IAM"
  type        = string
}

variable "mlflow_instance_type" {
  description = "EC2 instance type for MLflow server"
  type        = string
  default     = "t3.small"
}

variable "mlflow_key_name" {
  description = "Name of the EC2 key pair for SSH access. Leave empty for no SSH key."
  type        = string
  default     = ""
}

variable "mlflow_allowed_cidr" {
  description = "CIDR blocks allowed to access MLflow port 5000 and SSH port 22"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "wine_csv_local_path" {
  description = "Local path to wine.csv to upload to S3 on apply. Leave empty to skip."
  type        = string
  default     = ""
}
