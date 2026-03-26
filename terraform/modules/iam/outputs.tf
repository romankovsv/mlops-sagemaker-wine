output "github_role_arn" {
  value = aws_iam_role.github_actions.arn
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker.arn
}

output "mlflow_instance_profile_name" {
  value = aws_iam_instance_profile.mlflow_ec2.name
}
