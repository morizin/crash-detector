from pandas import DataFrame
from ...config.config_entity import DataTransformationConfig
from ...core import Directory
from ...errors import TransformationError
from sklearn.model_selection import KFold, StratifiedKFold
from ...utils.common import save_csv
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import LabelEncoder
from typeguard import typechecked
from ...config.config_entity import DataSchema
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


@typechecked
def split_dataset(
    config: DataTransformationConfig,
    data: pd.DataFrame,
    schema: DataSchema,
    filename: str,
    outdir: Directory | None = None,
) -> DataFrame:
    split_config = config.split
    dataname = schema.name

    data["fold"] = -1

    labels = schema.target

    if isinstance(labels, list):
        if len(labels) > 1:
            logger.error("Multi Label Stratified K-Fold is not implemented yet.")
            split_config.type = "kfold"
        else:
            labels = labels[0]
            split_config.type = "skfold"

    if split_config.type == "kfold":
        splitter = KFold
    elif split_config.type == "skfold":
        splitter = StratifiedKFold

    if hasattr(splitter, "labels"):
        labels = split_config.labels

    splitter = splitter(
        split_config.n_splits,
        shuffle=os.environ.get("PYTHONHASHSEED", 0) != 0,
        random_state=int(os.environ["PYTHONHASHSEED"])
        if hasattr(os.environ, "PYTHONHASHSEED")
        else None,
    )
    try:
        le_columns = []
        if labels:
            if isinstance(labels, list):
                for col in labels:
                    if not is_integer_dtype(data[col]):
                        logger.info(f"Label Encoding the dataset column '{col}'")
                        data[f"{col}_le"] = LabelEncoder().fit_transform(data[col])
                        col = f"{col}_le"
                    le_columns.append(col)

            elif isinstance(labels, str):
                le_columns = labels
                if not is_integer_dtype(data[labels]):
                    le_columns = f"{labels}_le"
                    data[le_columns] = LabelEncoder().fit_transform(data[labels])

            else:
                e = TransformationError(
                    "Labels are neither str or list[str]",
                    dataname=dataname,
                    file_name=filename,
                )
                logger.error(e)
                raise e

            logger.info(
                f"Folding '{dataname}.{filename}' into {split_config.n_splits} using {split_config.type} on column(s) {le_columns}"
            )

            for fold, (_, test_index) in enumerate(
                splitter.split(data, data[le_columns])
            ):
                data.loc[test_index, "fold"] = fold

            if isinstance(le_columns, str):
                le_columns = le_columns if le_columns.endswith("_le") else None
            elif isinstance(le_columns, list):
                le_columns = [col for col in le_columns if col.endswith("_le")]

            if le_columns:
                data = data.drop(le_columns, axis=1)

    except Exception as e:
        e = TransformationError(
            f"Labels are not given to use {split_config.type} folding technique. \nUsing Regular KFold",
            dataname=dataname,
            file_name=filename,
        )
        logger.error(e)

        if le_columns:
            if isinstance(le_columns, list):
                data = data.drop(
                    [col for col in le_columns if col.endswith("_le")], axis=1
                )
            elif le_columns.endswith("_le"):
                data = data.drop(le_columns, axis=1)

        splitter = KFold(
            split_config.n_splits,
            shuffle=os.environ.get("PYTHONHASHSEED", 0) != 0,
            random_state=int(os.environ.get("PYTHONHASHSEED", 1234)),
        )
        logger.info(
            f"Folding '{dataname}.{filename}' into {split_config.n_splits} using kfold"
        )

        for fold, (_, test_index) in enumerate(splitter.split(data)):
            data.loc[test_index, "fold"] = fold

    if outdir:
        save_csv(data, outdir // f"folded_{dataname}" / filename)

    return data
