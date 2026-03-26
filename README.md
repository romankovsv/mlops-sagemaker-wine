MLOps Sagemaker MLFlow FastApi Terraform


![alt text](image.png)

MLFlow version 2.21

starting on EC2

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://{{AWS_S3_BUCKET}}/mlflow/



To start FastApi server with Sagemaker endpoint backend

cd ml-api
pip install -r requirements.txt

# Set AWS credentials (needs sagemaker:InvokeEndpoint permission)
set region, aws access key and secret key

uvicorn app:app --host 0.0.0.0 --port 8000