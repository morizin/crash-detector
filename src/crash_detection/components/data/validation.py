import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from ...config.config_entity import DataValidataionConfig
from ...config.artifact_entity import DataValidationArtifact
from ...utils.common import save_json, save_csv
import logging

logger = logging.getLogger(__name__)


tqdm.pandas()


def load_csv(file_path: Path | str) -> pd.DataFrame:
    return pd.read_csv(file_path)


class DataValidationComponent:
    def __init__(self, config: DataValidataionConfig):
        self.config = config
        tree = lambda: defaultdict(tree)
        self.status = tree()
        self.status["validation_status"] = True

    def validate(self) -> DataValidationArtifact:
        schemas = deepcopy(self.config.schemas)
        for schema_name, schema in schemas.items():
            logger.info(f"Validating dataset: {schema_name}")
            train_files = []
            test_files = []

            for file in schema.train + schema.test:
                basename = str(os.path.basename(file))
                logger.info(f"Validating file: {basename}")
                if os.path.exists(file):
                    df = load_csv(file)
                elif not os.path.exists(os.path.join(schema.path, file)):
                    df = load_csv(os.path.join(schema.path, file))
                else:
                    logger.error(f"File {file} not found for dataset {schema_name}")
                    self.status[schema_name][basename] = {
                        "schema_validation": False,
                        "additional_invalid_schema_column": [],
                        "missing_column": True,
                        "additional_missing_column": [],
                        "missing_image": False,
                    }
                    self.status["validation_status"] = False
                    continue

                logger.info(f"Loaded data shape: {df.shape}")

                self.status[schema_name][basename] = {
                    "schema_validation": True,
                    "additional_invalid_schema_column": [],
                    "missing_column": False,
                    "additional_missing_column": [],
                    "missing_image": False,
                }
                for col, dtype in schema.columns.items():
                    if col in df.columns:
                        if not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
                            logger.error(
                                f"Column {col} has dtype {df[col].dtype}, expected {dtype}"
                            )
                            self.status[schema_name][basename]["schema_validation"] = (
                                False
                            )
                            self.status["validation_status"] = False

                    else:
                        logger.error(f"Column {col} is missing in the dataset")
                        self.status[schema_name][basename]["missing_column"] = True
                        self.status["validation_status"] = False

                for col, dtype in schema.additional_properties.items():
                    if col in df.columns:
                        if not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
                            logger.warning(
                                f"Additional column {col} has dtype {df[col].dtype}, expected {dtype}"
                            )
                            self.status[schema_name][basename][
                                "additional_invalid_schema_column"
                            ].append(col)
                            self.status["validation_status"] = False

                    else:
                        logger.warning(
                            f"Additional column {col} is missing in the dataset"
                        )
                        self.status[schema_name][basename][
                            "additional_missing_column"
                        ].append(col)

                # Remove undefined columns
                for col in self.status[schema_name][basename][
                    "additional_invalid_schema_column"
                ]:
                    logger.warning(
                        f"Dropping additional column {col} due to invalid schema"
                    )
                    df.drop(columns=[col], inplace=True)

                for col in df.columns:
                    if (
                        col not in schema.columns
                        and col not in schema.additional_properties
                        and col != schema.target
                    ):
                        logger.warning(
                            f"Dropping column {col} since not defined in the schema"
                        )
                        df.drop(columns=[col], inplace=True)
                        self.status[schema_name][basename][
                            "additional_missing_column"
                        ].append(col)

                train = True if file in schema.train else False

                image_folder = (
                    schema.train_image_folder if train else schema.test_image_folder
                )
                if not image_folder.exists():
                    image_folder = schema.path // image_folder

                df["filename"] = df["filename"].progress_apply(
                    lambda x: str(image_folder / x)
                )

                missing_images = df["filename"].progress_apply(
                    lambda x: not os.path.exists(x)
                )

                if missing_images.any():
                    logger.error(
                        f"{missing_images.sum()} Missing images found in dataset {schema_name}, file {os.path.basename(file)}"
                    )
                    valid_df = df[~missing_images].reset_index(drop=True)
                    invalid_df = df[missing_images].reset_index(drop=True)
                    self.status[schema_name][basename]["missing_image"] = True
                    self.status["validation_status"] = False
                else:
                    valid_df = df
                    invalid_df = pd.DataFrame(columns=df.columns)

                save_csv(
                    path=self.config.outdir // "valid" // schema_name / basename,
                    data=valid_df,
                )

                if train:
                    schema.train = [f for f in schema.train if f != file]
                logger.info(
                    f"Saved validated data to {self.config.outdir // 'valid' // schema_name / basename}"
                )

                if missing_images.any():
                    save_csv(
                        path=self.config.outdir // "invalid" // schema_name / basename,
                        data=invalid_df,
                    )
                    logger.info(
                        f"Saved invalid data to {self.config.outdir // 'invalid' // schema_name / basename}"
                    )

                eval(f"{'train' if train else 'test'}_files").append(
                    self.config.outdir // "valid" // schema_name / basename
                )

            schema.train = train_files
            schema.test = test_files

        save_json(self.config.report_path, self.status)
        logger.info(f"Saved validation report to {self.config.report_path}")

        return DataValidationArtifact(
            report_path=self.config.report_path,
            image_dir=self.config.indir,
            valid_data_dir=self.config.outdir // "valid",
            invalid_data_dir=self.config.outdir // "invalid",
            schemas=schemas,
            is_validated=self.status["validation_status"],
        )

    def __call__(self) -> DataValidationArtifact:
        return self.validate()
