from pydantic import BaseModel
from pathlib import Path
from typing import Optional


class DataSource(BaseModel):
    id: str
    source: str
    type: Optional[str] = None
    link: str


class DataIngestionConfig(BaseModel):
    data_sources: dict[str, DataSource]
    outdir: Path
