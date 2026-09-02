"""Thin S3-compatible object storage wrapper. Works against real AWS S3
(leave s3_endpoint_url unset) or MinIO for local dev (point it at the
compose service). All video/frame/calibration/stats bytes live here, never
on a shared local disk - the API and worker processes don't need to share a
filesystem.
"""

import boto3
from botocore.client import Config

from app.config import get_settings


def _client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket() -> None:
    settings = get_settings()
    client = _client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except Exception:
        client.create_bucket(Bucket=settings.s3_bucket)


def upload_file(local_path: str, key: str, content_type: str | None = None) -> None:
    settings = get_settings()
    extra_args = {"ContentType": content_type} if content_type else {}
    _client().upload_file(local_path, settings.s3_bucket, key, ExtraArgs=extra_args)


def upload_bytes(data: bytes, key: str, content_type: str | None = None) -> None:
    settings = get_settings()
    extra_args = {"ContentType": content_type} if content_type else {}
    _client().put_object(Bucket=settings.s3_bucket, Key=key, Body=data, **extra_args)


def download_file(key: str, local_path: str) -> None:
    settings = get_settings()
    _client().download_file(settings.s3_bucket, key, local_path)


def download_bytes(key: str) -> bytes:
    settings = get_settings()
    obj = _client().get_object(Bucket=settings.s3_bucket, Key=key)
    return obj["Body"].read()


def presigned_url(key: str, expires_seconds: int = 3600) -> str:
    settings = get_settings()
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_public_endpoint_url or settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4"),
    )
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )
