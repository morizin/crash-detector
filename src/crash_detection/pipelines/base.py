from ..config import ConfigurationManager
from ..core.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
)
from ..core.artifact_entity import DataIngestionArtifact, DataTransformationArtifact

from typeguard import typechecked
from .. import logger

from ..components.data.ingestion import DataIngestionComponent
from ..components.data.transformation import DataTransformationComponent


class BasePipeline:
    def __init__(self):
        try:
            logger.info("Configuring....")
            self.config: ConfigurationManager = ConfigurationManager()
        except Exception as e:
            logger.error(f"Error Configuring... {e}")

    @typechecked
    def do_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Data Ingestion ....")
            data_ingestion_config: DataIngestionConfig = (
                self.config.get_data_ingestion_config()
            )

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionComponent(
                data_ingestion_config
            )()

            logger.info("Data Ingesion Completed")
            return data_ingestion_artifact

        except Exception as e:
            logger.error(f"Error during data ingestion {e}")
            raise e

    @typechecked
    def do_data_transformation(
        self, data_transformation_config: DataTransformationConfig
    ) -> DataTransformationArtifact | None:
        try:
            logger.info("Data Transformation ....")
            data_transformation_component = DataTransformationComponent(
                config=data_transformation_config
            )()
            logger.info("Data Transformation Completed")
            return data_transformation_component
        except Exception as e:
            logger.error(f"Error during data transformation {e}")
            raise e

    @typechecked
    def do_model_trainer(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            logger.info("Model Trainer ....")
            model_trainer_config: dict[str, ModelTrainingConfig] = (
                self.config.model_training_config(
                    data_ingestion_artifact=data_ingestion_artifact
                )
            )

            for model, config in model_trainer_config.items():
                data_transformation_artifact: DataTransformationArtifact = (
                    self.do_data_transformation(config.transforms)
                )

            # model_trainer_component = ModelTrainerComponent(
            #     config=model_trainer_config,
            #     data_transformation_artifact=data_transformation_artifact,
            # )()
            logger.info("Model Trainer Completed")
            # return model_trainer_component
        except Exception as e:
            logger.error(f"Error during model trainer {e}")
            raise e

    @typechecked
    def kickoff(
        self,
    ):
        logger.info("Kicking off Base Pipeline")
        data_ingestion_artifact: DataIngestionArtifact = self.do_data_ingestion()

        self.do_model_trainer(
            data_ingestion_artifact=data_ingestion_artifact,
        )
        logger.info("Base Pipeline Completed")
        print(data_ingestion_artifact)
        return data_ingestion_artifact
