import os
from pathlib import Path
from typeguard import typechecked
from typing import Union
from pydantic import BaseModel, field_validator
from .. import logger


class Directory(BaseModel):
    path: Path

    @field_validator("path", mode="before")
    @typechecked
    def is_directory(cls, path: Path | str) -> Path:
        if isinstance(path, str):
            path = Path(path)
        path = path.expanduser()
        Directory.create(path)
        return path

    @typechecked
    def __floordiv__(self, other: Union["Directory", Path, str]) -> "Directory":
        if isinstance(other, str) or isinstance(other, Path):
            path = self.path / other
        if isinstance(other, Directory):
            path = self.path / other.path

        return Directory(path=path)

    @typechecked
    def __truediv__(self, other: Union["Directory", Path, str]) -> Path:
        if isinstance(other, str) or isinstance(other, Path):
            path = self.path / other
        if isinstance(other, Directory):
            path = self.path / other.path
        return path

    @staticmethod
    def create(path: Path | str) -> None:
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            logger.info(f"Creating Directory {str(path)}")
            path.mkdir(parents=True, exist_ok=True)

    def __str__(
        self,
    ):
        return self.path.as_posix()

    def listdir(self, *args: tuple[str]) -> list[str]:
        path: Path = self.path
        if args is not None:
            for fold in args:
                path /= fold
                if not path.exists():
                    e = FileNotFoundError(path)
                    logger.error(e)
                    raise e
        return list(map(lambda x: os.path.basename(x.as_posix()), self.path.iterdir()))

    def exists(
        self,
    ) -> bool:
        return self.path.exists()
