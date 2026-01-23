import kaggle
import os
from ...config.config_entity import DataIngestionConfig, DataSchema
from ...config.artifact_entity import DataIngestionArtifact
from ...errors import ComponentError
from ... import logger


class DataIngestionComponent:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.cfg: DataIngestionConfig = data_ingestion_config
            self.kaggle_api = kaggle.KaggleApi()
        except Exception as e:
            logger.error(f"Error at Data Ingestion Component {e}")

    def __call__(self):
        for name, source in self.cfg.data_sources.items():
            target_path = self.cfg.outdir / name
            if (self.cfg.outdir / name).exists() and len(os.listdir(target_path)):
                logger.info(f"Dataset {name} already exists at {target_path}")
                continue
            match source.source:
                case "kaggle":
                    match source.type:
                        case "datasets":
                            logger.info(
                                f"Downloading dataset {name} : {source.source}::{source.type}::{source.link} at {target_path}"
                            )
                            self.kaggle_api.dataset_download_cli(
                                source.link,
                                path=target_path,
                                unzip=True,
                            )
                        case _:
                            e = ComponentError(
                                self, f"Source having type {source.type} not found"
                            )
                            logger.error(e)
                            raise e
                case _:
                    e = ComponentError(self, f"Source {source.source} not found")
                    logger.error(e)
                    raise e

        return DataIngestionArtifact(
            path=self.cfg.outdir,
            schemas={
                name: DataSchema(name=name, path=self.cfg.outdir / name)
                for name in self.cfg.data_sources.keys()
            },
        )
