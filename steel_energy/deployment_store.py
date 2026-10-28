import os
from functools import lru_cache
from io import BytesIO

from gridfs import GridFSBucket
from pymongo import MongoClient


@lru_cache(maxsize=1)
def _database():
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("Set the MONGODB_URI environment variable to enable persistent model storage.")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client["steel_energy"]


def load_active_version() -> dict[str, object] | None:
    record = _database()["deployment_state"].find_one({"_id": "active_model"})
    return record


def download_artifact(pathname: str) -> bytes:
    bucket = GridFSBucket(_database(), bucket_name="deployment_artifacts")
    files = list(
        bucket.find({"filename": pathname})
        .sort("uploadDate", -1)
        .limit(1)
    )
    if not files:
        raise FileNotFoundError(f"Deployment artifact {pathname!r} is missing from MongoDB.")
    with bucket.open_download_stream(files[0]._id) as stream:
        return stream.read()


def publish_version(version: str, artifacts: dict[str, bytes]) -> None:
    database = _database()
    bucket = GridFSBucket(database, bucket_name="deployment_artifacts")
    for pathname, content in artifacts.items():
        bucket.upload_from_stream(
            pathname,
            BytesIO(content),
            metadata={"model_version": version},
        )
    database["deployment_state"].update_one(
        {"_id": "active_model"},
        {"$set": {"model_version": version, "artifacts": list(artifacts)}},
        upsert=True,
    )
