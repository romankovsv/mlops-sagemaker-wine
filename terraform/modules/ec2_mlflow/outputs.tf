output "public_ip" {
  description = "Public IP of the MLflow EC2 instance"
  value       = aws_instance.mlflow.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.mlflow.id
}
