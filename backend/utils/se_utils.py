import boto3
from botocore.client import Config
from uuid import uuid4

S3_BUCKET = "city-surveillance"
S3_ENDPOINT = "http://localhost:9000"  # MinIO local
S3_ACCESS_KEY = "minioadmin"
S3_SECRET_KEY = "minioadmin"

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1"
)

def upload_to_s3(file_obj, filename=None):
    key = filename or f"{uuid4()}.jpg"
    s3.upload_fileobj(file_obj, S3_BUCKET, key)
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"
