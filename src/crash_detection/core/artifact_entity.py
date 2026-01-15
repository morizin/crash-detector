from pydantic import BaseModel
from pathlib import Path


class DataIngestionArtifact(BaseModel):
    names: list[str]
    path: Path
