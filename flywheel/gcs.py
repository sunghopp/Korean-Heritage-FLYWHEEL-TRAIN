from __future__ import annotations

import json
from typing import Any

from google.cloud import storage


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    bucket, _, name = uri[5:].partition("/")
    return bucket, name


def gs_uri(bucket: str, name: str) -> str:
    return f"gs://{bucket}/{name.lstrip('/')}"


class GCS:
    def __init__(self, project: str, bucket_name: str):
        self.client = storage.Client(project=project)
        self.bucket = self.client.bucket(bucket_name)

    def read_json(self, name: str) -> tuple[dict[str, Any], int]:
        blob = self.bucket.blob(name)
        blob.reload()
        return json.loads(blob.download_as_text()), int(blob.generation)

    def write_json(self, name: str, value: dict[str, Any], *, generation: int | None = None, create_only: bool = False) -> None:
        kwargs: dict[str, Any] = {}
        if create_only:
            kwargs["if_generation_match"] = 0
        elif generation is not None:
            kwargs["if_generation_match"] = generation
        self.bucket.blob(name).upload_from_string(
            json.dumps(value, ensure_ascii=False, indent=2), content_type="application/json", **kwargs
        )

    def write_text(self, name: str, value: str, *, create_only: bool = False) -> None:
        kwargs: dict[str, Any] = {"if_generation_match": 0} if create_only else {}
        self.bucket.blob(name).upload_from_string(value, content_type="application/x-ndjson", **kwargs)

    def download_uri(self, uri: str) -> str:
        bucket, name = parse_gs_uri(uri)
        return self.client.bucket(bucket).blob(name).download_as_text()

    def upload_uri(self, uri: str, local_path: str) -> None:
        bucket, name = parse_gs_uri(uri)
        self.client.bucket(bucket).blob(name).upload_from_filename(local_path)

    def copy_prefix(self, source_uri: str, destination_uri: str, *, overwrite: bool = True) -> list[str]:
        """Copy every object below a GCS prefix and return relative file names."""
        source_bucket_name, source_prefix = parse_gs_uri(source_uri)
        destination_bucket_name, destination_prefix = parse_gs_uri(destination_uri)
        source_prefix = source_prefix.rstrip("/") + "/"
        destination_prefix = destination_prefix.rstrip("/") + "/"
        source_bucket = self.client.bucket(source_bucket_name)
        destination_bucket = self.client.bucket(destination_bucket_name)
        copied: list[str] = []
        for blob in source_bucket.list_blobs(prefix=source_prefix):
            if blob.name.endswith("/"):
                continue
            relative = blob.name.removeprefix(source_prefix)
            destination_name = f"{destination_prefix}{relative}"
            if not overwrite and destination_bucket.blob(destination_name).exists():
                raise FileExistsError(f"Backup already exists: gs://{destination_bucket_name}/{destination_name}")
            source_bucket.copy_blob(blob, destination_bucket, destination_name)
            copied.append(relative)
        if not copied:
            raise FileNotFoundError(f"No objects found below {source_uri}")
        return copied

    def delete_prefix_except(self, prefix_uri: str, allowed_relative_names: set[str]) -> None:
        """Remove stale files only after a complete replacement prefix exists."""
        bucket_name, prefix = parse_gs_uri(prefix_uri)
        prefix = prefix.rstrip("/") + "/"
        bucket = self.client.bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=prefix):
            relative = blob.name.removeprefix(prefix)
            if relative and relative not in allowed_relative_names:
                blob.delete()
