resource "aws_s3_bucket" "main" {
  bucket        = "${var.project}-${var.account_id}-${var.env}"
  force_destroy = var.env == "dev"
}

resource "aws_s3_bucket_versioning" "main" {
  bucket = aws_s3_bucket.main.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "main" {
  bucket = aws_s3_bucket.main.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "main" {
  bucket                  = aws_s3_bucket.main.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Optional: upload wine.csv on apply
resource "aws_s3_object" "wine_csv" {
  count  = var.wine_csv_local_path != "" ? 1 : 0
  bucket = aws_s3_bucket.main.id
  key    = "data/wine.csv"
  source = var.wine_csv_local_path != "" ? var.wine_csv_local_path : "/dev/null"
  etag   = var.wine_csv_local_path != "" ? filemd5(var.wine_csv_local_path) : null
}
