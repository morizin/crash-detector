import torch
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np
from ...config.config_entity import ModelTrainingConfig
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


def get_dataset(
    config: ModelTrainingConfig,
) -> Dataset:
    if config.type == "video-2d-classification":
        return VideoCrashDataset(
            data=config.data,
            img_dir=config.img_dir,
            cache_dir=config.cache_dir,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {config.type}")


class CachedVideoCrashDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.filenames = data.filename.values
        self.targets = data.target.values

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        filename = self.filenames[idx]
        video_tensor = np.load(filename)

        return video_tensor, torch.tensor(self.targets[idx], dtype=torch.long)


class VideoCrashDataset(Dataset):
    def __init__(self, data: pd.DataFrame, img_dir: Path | str):
        self.data = data
        self.vids = data.vid[::20].values
        self.targets = data.target[::20].values
        self.img_dir = Path(img_dir)

    def _load_video(self, vid):
        video_tensor = np.zeros((224, 224, 20), dtype=np.uint8)

        for i in range(20):
            path = self.img_dir / f"{str(vid).zfill(5)}_{str(i).zfill(2)}.jpg"

            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (224, 224))
            video_tensor[:, :, i] = image

        video_tensor = torch.from_numpy(video_tensor)
        return video_tensor

    def __len__(self) -> int:
        return len(self.vids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        vid = self.vids[idx]
        video_tensor = self._load_video(vid)

        return video_tensor, torch.tensor(self.targets[idx], dtype=torch.long)
