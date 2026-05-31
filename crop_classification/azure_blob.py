import io
from azure.storage.blob import BlobServiceClient

try:
    from config import CONNECTION_STRING, CONTAINER_NAME
except ImportError:
    raise ImportError(
        "config.py not found. Copy crop_classification/config_template.py to config.py and fill in Azure credentials."
    )


def get_container_client():
    """Create and return an Azure Blob Storage container client."""
    service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = service_client.get_container_client(CONTAINER_NAME)
    if not container_client.exists():
        container_client.create_container()
    return container_client


def download_blob_to_bytes(blob_name):
    """Download a blob and return its content as bytes."""
    client = get_container_client().get_blob_client(blob_name)
    stream = io.BytesIO()
    download_stream = client.download_blob()
    download_stream.readinto(stream)
    stream.seek(0)
    return stream


def download_blob_to_path(blob_name, local_path):
    """Download a blob directly to a local file path."""
    client = get_container_client().get_blob_client(blob_name)
    with open(local_path, "wb") as handle:
        handle.write(client.download_blob().readall())


def upload_bytes(blob_name, data, overwrite=True):
    """Upload raw bytes or a file-like object to Azure Blob Storage."""
    client = get_container_client().get_blob_client(blob_name)
    client.upload_blob(data, overwrite=overwrite)


def upload_local_file(local_path, blob_name, overwrite=True):
    """Upload a local file to Azure Blob Storage."""
    client = get_container_client().get_blob_client(blob_name)
    with open(local_path, "rb") as data:
        client.upload_blob(data, overwrite=overwrite)
