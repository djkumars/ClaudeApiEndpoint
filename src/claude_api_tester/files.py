"""Helpers for the Claude Files API."""

from pathlib import Path

from anthropic import Anthropic

FILES_API_BETA = ["files-api-2025-04-14"]


def upload_file(client: Anthropic, file_path: str | Path):
    """Upload a local file and return its metadata."""
    path = Path(file_path)
    with path.open("rb") as file_handle:
        return client.beta.files.upload(file=file_handle, betas=FILES_API_BETA)


def list_files(client: Anthropic, limit: int = 20):
    """List uploaded files."""
    return client.beta.files.list(limit=limit, betas=FILES_API_BETA)


def retrieve_file_metadata(client: Anthropic, file_id: str):
    """Retrieve metadata for one uploaded file."""
    return client.beta.files.retrieve_metadata(file_id, betas=FILES_API_BETA)


def download_file(
    client: Anthropic,
    file_id: str,
    destination: str | Path,
) -> Path:
    """Download a file to a local destination."""
    path = Path(destination)
    response = client.beta.files.download(file_id, betas=FILES_API_BETA)
    response.write_to_file(path)
    return path


def delete_file(client: Anthropic, file_id: str):
    """Delete an uploaded file."""
    return client.beta.files.delete(file_id, betas=FILES_API_BETA)
