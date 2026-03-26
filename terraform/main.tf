data "aws_caller_identity" "current" {}

locals {
  env        = terraform.workspace
  account_id = data.aws_caller_identity.current.account_id
}

module "s3_bucket" {
  source = "./modules/s3_bucket"

  project             = var.project
  env                 = local.env
  account_id          = local.account_id
  wine_csv_local_path = var.wine_csv_local_path
}

module "iam" {
  source = "./modules/iam"

  project                  = var.project
  env                      = local.env
  bucket_arn               = module.s3_bucket.bucket_arn
  bucket_name              = module.s3_bucket.bucket_name
  github_org               = var.github_org
  github_repo              = var.github_repo
  github_oidc_provider_arn = var.github_oidc_provider_arn
}

module "ec2_mlflow" {
  source = "./modules/ec2_mlflow"

  project                      = var.project
  env                          = local.env
  instance_type                = var.mlflow_instance_type
  key_name                     = var.mlflow_key_name
  allowed_cidr                 = var.mlflow_allowed_cidr
  bucket_name                  = module.s3_bucket.bucket_name
  mlflow_instance_profile_name = module.iam.mlflow_instance_profile_name
}
