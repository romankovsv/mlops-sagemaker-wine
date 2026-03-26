data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "mlflow" {
  name        = "${var.project}-${var.env}-mlflow-sg"
  description = "MLflow server: port 5000 and SSH"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "MLflow UI"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "mlflow" {
  ami                         = data.aws_ami.amazon_linux_2023.id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.mlflow.id]
  iam_instance_profile        = var.mlflow_instance_profile_name
  associate_public_ip_address = true
  key_name                    = var.key_name != "" ? var.key_name : null

  # Rendered at plan time — substitutes ${bucket_name} with actual bucket name
  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    bucket_name = var.bucket_name
  })

  # Replace EC2 instance when user_data changes
  user_data_replace_on_change = true

  root_block_device {
    volume_size           = 20
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  tags = {
    Name = "${var.project}-${var.env}-mlflow"
  }
}
