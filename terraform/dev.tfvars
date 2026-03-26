aws_region  = "us-east-1"
project     = "mlops-wine"
github_org  = "romankovsv" # REPLACE if different
github_repo = "mlops-sagemaker-wine"

# Get this from: AWS Console -> IAM -> Identity providers -> token.actions.githubusercontent.com -> ARN
github_oidc_provider_arn = "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com" # REPLACE ACCOUNT_ID

mlflow_instance_type = "t3.small"
mlflow_key_name      = "" # REPLACE with your EC2 key pair name, or leave empty
mlflow_allowed_cidr  = ["0.0.0.0/0"]

# Set to local path of wine.csv to upload it on apply, e.g. "../data/wine.csv"
# Leave empty if already uploaded to S3
wine_csv_local_path = ""
