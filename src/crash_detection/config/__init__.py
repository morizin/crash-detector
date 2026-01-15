from pathlib import Path
from ..core.config_entity import DataSource, DataIngestionConfig
from ..utils.common import load_yaml, create_directories
from ..constants import CONFIG_PATH, DATA_DIRECTORY_NAME, SCHEMA_DIR
import os


class ConfigManager:
    def __init__(self, config_path=CONFIG_PATH):
        self.config = load_yaml(CONFIG_PATH)
        self.artifact_path = Path(self.config.artifact_path)
        self.schema_path = SCHEMA_DIR

        create_directories(self.artifact_path)

    def get_data_sources(self) -> dict[str, DataSource]:
        data_sources_v = dict()
        for name, source in self.config.data_sources.items():
            data_sources_v[name] = DataSource(
                id=name,
                source=source.source,
                type=source.type if hasattr(source, "type") else None,
                link=source.link,
            )
        return data_sources_v

    def get_data_ingestion_config(
        self,
    ) -> DataIngestionConfig:
        outdir = self.artifact_path / DATA_DIRECTORY_NAME
        create_directories(outdir)

        return DataIngestionConfig(data_sources=self.get_data_sources(), outdir=outdir)
