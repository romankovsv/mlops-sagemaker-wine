aws_region  = "us-east-1"
project     = "mlops-wine"
github_org  = "romankovsv" # REPLACE if different
github_repo = "mlops-sagemaker-wine"

github_oidc_provider_arn = "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com" # REPLACE ACCOUNT_ID

mlflow_instance_type = "t3.medium"
mlflow_key_name      = ""            # REPLACE with your EC2 key pair name
mlflow_allowed_cidr  = ["0.0.0.0/0"] # RECOMMENDED: restrict to your IP ["1.2.3.4/32"]

# Dataset already in S3 for prod — leave empty
wine_csv_local_path = ""
