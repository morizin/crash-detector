from ...config.config_entity import DataTransformationConfig, DataSchema
from ...config.artifact_entity import DataTransformationArtifact
from ...utils.common import load_csv, save_csv
from ...core import Directory
import pandas as pd
from typeguard import typechecked
from tqdm import tqdm
from copy import deepcopy
from functools import partial
from .utils import split_dataset
from concurrent.futures import ThreadPoolExecutor
import cv2
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataTransformationComponent:
    @typechecked
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    @typechecked
    def _cache_video(self, vid: int, indir: Directory, outdir: Directory) -> None:
        cache_file = outdir / f"{str(vid).zfill(5)}.npy"
        if cache_file.exists():
            return

        video_tensor = np.zeros((224, 224, 20), dtype=np.uint8)
        for i in range(20):
            path = indir / f"{str(vid).zfill(5)}_{str(i).zfill(2)}.jpg"

            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (224, 224))
            video_tensor[:, :, i] = image

        np.save(cache_file, video_tensor)

    @typechecked
    def transform(
        self, data: pd.DataFrame, indir: Directory, outdir: Directory
    ) -> bool:
        vid_list = data["vid"].unique().tolist()

        cache_func = partial(self._cache_video, indir=indir, outdir=outdir)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(
                tqdm(
                    pool.map(cache_func, vid_list),
                    desc="Parallel Cache Input",
                    total=len(vid_list),
                )
            )
        return True

    @typechecked
    def __call__(self) -> DataTransformationArtifact:
        schemas: dict[str, DataSchema] = deepcopy(self.config.schemas)
        if self.config.indir.exists():
            train_datas = []
            valid_datas = []
            test_datas = []
        else:
            e = FileNotFoundError(f"Data path {self.config.indir} does not exist.")
            logger.error(e)
            raise e

        for name, schema in schemas.items():
            for split in ["train", "test"]:
                for filename in eval(f"schema.{split}"):
                    if (filename).exists():
                        data = load_csv(filename)
                    elif (self.config.indir / name / filename).exists():
                        data = load_csv(self.config.indir / name / filename)
                    else:
                        e = FileNotFoundError(f"Data file {filename} does not exist.")
                        logger.error(e)
                        raise e

                    data = data[["vid", schema.target]].drop_duplicates(
                        ignore_index=True
                    )
                    self.transform(
                        data,
                        schema.path // getattr(schema, f"{split}_image_folder"),
                        self.config.outdir
                        // name
                        // getattr(schema, f"{split}_image_folder"),
                    )

                    data["filename"] = data.apply(
                        lambda row: self.config.outdir
                        // name
                        // getattr(schema, f"{split}_image_folder")
                        / f"{str(row['vid']).zfill(5)}.npy",
                        axis=1,
                    )

                    if split == "train":
                        data = split_dataset(
                            config=self.config,
                            data=data,
                            schema=schema,
                            filename=os.path.basename(filename),
                            outdir=None,
                        )
                        train_folds = int(
                            self.config.split.n_splits * self.config.split.ratio
                        )

                        train_data = data[data["fold"] < train_folds].reset_index(
                            drop=True
                        )[["vid", schema.target, "filename"]]

                        save_csv(self.config.outdir / name / f"{split}.csv", data=data)

                        valid_data = data[data["fold"] >= train_folds].reset_index(
                            drop=True
                        )[["vid", schema.target, "filename"]]

                        train_datas.append(train_data)
                        if not valid_data.empty:
                            valid_datas.append(valid_data)
                            save_csv(
                                self.config.outdir / name / "valid.csv", data=valid_data
                            )
                    else:
                        test_datas.append(data)
                        save_csv(self.config.outdir / name / f"{split}.csv", data=data)

            schema.path = self.config.outdir / name
            schema.train = [self.config.outdir / name / "train.csv"]
            schema.valid = [self.config.outdir / name / "valid.csv"]
            schema.valid_image_folder = schema.train_image_folder
            schema.test = [self.config.outdir / name / "test.csv"]

        train_data = pd.concat(train_datas).reset_index(drop=True)
        valid_data = pd.concat(valid_datas).reset_index(drop=True)
        test_data = pd.concat(test_datas).reset_index(drop=True)

        save_csv(self.config.outdir / "train.csv", data=train_data)
        save_csv(self.config.outdir / "valid.csv", data=valid_data)
        save_csv(self.config.outdir / "test.csv", data=test_data)

        return DataTransformationArtifact(
            path=self.config.outdir,
            train_file_path=self.config.outdir / "train.csv",
            valid_file_path=self.config.outdir / "valid.csv",
            test_file_path=self.config.outdir / "test.csv",
            schemas=schemas,
        )


if __name__ == "__main__":
    pass
    # for model, config in model_training_config.items():
    #     data_transformation_component = DataTransformationComponent(config=config.transforms)
    #     data_transformation_component(config.datasets)
