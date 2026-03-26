 Mlops-aws-sagemaker-end-to-end


![alt text](image.png)

MLFlow version 2.21

starting on EC2

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://{{AWS_S3_BUCKET}}/mlflow/
