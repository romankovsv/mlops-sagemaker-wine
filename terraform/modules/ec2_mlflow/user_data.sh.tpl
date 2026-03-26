#!/bin/bash
set -e

# Update and install dependencies
yum update -y
yum install -y python3 python3-pip sqlite

# Install MLflow and boto3 (boto3 needed for S3 artifact store)
pip3 install mlflow==2.21.0 boto3

# Create working directory
mkdir -p /opt/mlflow
chown ec2-user:ec2-user /opt/mlflow

# Write systemd service unit
cat > /etc/systemd/system/mlflow.service << UNIT
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/mlflow
ExecStart=/usr/local/bin/mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////opt/mlflow/mlflow.db --default-artifact-root s3://${bucket_name}/mlflow/
Restart=always
RestartSec=10
Environment=HOME=/home/ec2-user

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable mlflow
systemctl start mlflow
