variable "project" {
  type = string
}

variable "env" {
  type = string
}

variable "instance_type" {
  type    = string
  default = "t3.small"
}

variable "key_name" {
  type    = string
  default = ""
}

variable "allowed_cidr" {
  type    = list(string)
  default = ["0.0.0.0/0"]
}

variable "bucket_name" {
  type = string
}

variable "mlflow_instance_profile_name" {
  type = string
}
