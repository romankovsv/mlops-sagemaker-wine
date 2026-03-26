terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # BEFORE FIRST RUN: create this bucket manually once:
  #   aws s3 mb s3://YOUR-STATE-BUCKET --region us-east-1
  backend "s3" {
    bucket = "your-tf-state-bucket" # REPLACE with your state bucket name
    key    = "mlops-wine/terraform.tfstate"
    region = "us-east-1"
    # With workspaces, state is stored at:
    #   env:/dev/mlops-wine/terraform.tfstate
    #   env:/prod/mlops-wine/terraform.tfstate
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project
      Environment = terraform.workspace
      ManagedBy   = "terraform"
    }
  }
}
