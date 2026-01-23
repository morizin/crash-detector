import torch
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


class VideoCrashDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        img_dir: Path | str,
        cache_dir: Optional[Path | str] = None,
    ):
        self.data = data
        self.vids = data.vid[::20].values
        self.targets = data.target[::20].values
        self.img_dir = Path(img_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._parallel_build_cache()

    def _save_cache(self, vid):
        cache_file = self.cache_dir / f"vid_{vid}.pt"
        if not cache_file.exists():
            video_tensor = self._load_video(vid)
            torch.save(video_tensor, cache_file)

    def _build_cache(
        self,
    ):
        for vid in tqdm(self.vids, desc="Cacheing inputs"):
            self._save_cache(vid)
        return True

    def _parallel_build_cache(
        self,
    ):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(
                tqdm(
                    pool.map(self._save_cache, self.vids),
                    desc="Parallel Cache Input",
                    total=len(self.vids),
                )
            )
        return True

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
        if self.cache_dir and (self.cache_dir / f"vid_{vid}.pt").exists():
            video_tensor = torch.load(self.cache_dir / f"vid_{vid}.pt")
        else:
            video_tensor = self._load_video(vid)

        return video_tensor, torch.tensor(self.targets[idx], dtype=torch.long)
