import torch
from .model import Model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .optimizer import get_optimizer
from ...config.config_entity import ModelTrainingConfig
from ...config.artifact_entity import ModelTrainingArtifact
from typeguard import typechecked
from ...components.training.dataset import CachedVideoCrashDataset
from ...config.artifact_entity import DataTransformationArtifact
from ...utils.common import load_csv


class ModelTrainingComponent:
    @typechecked
    def __init__(
        self,
        config: ModelTrainingConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if self.device.type == "cuda":
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

        train_dataset, valid_dataset = self.get_data()
        train_dataset = CachedVideoCrashDataset(train_dataset)
        valid_dataset = (
            CachedVideoCrashDataset(valid_dataset) if not valid_dataset.empty else None
        )

        self.model = Model(config=self.config)

        self.model.to(self.device)

        self.train_batch_size = self.config.train_batch_size
        self.train_dataloader = self._create_dataloader(
            train_dataset, self.train_batch_size
        )
        self.n_train_batches = len(self.train_dataloader)

        if valid_dataset:
            self.valid_batch_size = self.config.valid_batch_size
            self.valid_dataloader = self._create_dataloader(
                valid_dataset, self.valid_batch_size
            )
            self.n_valid_batches = len(self.valid_dataloader)
        else:
            self.valid_dataloader = self.n_valid_batches = self.valid_batch_size = None

        self.optimizer = get_optimizer(
            self.config,
            self.model.parameters(),
        )

        self.i_epoch = 0

    def get_data(self):
        # Load CSV Files
        train_df = load_csv(self.data_transformation_artifact.train_file_path)
        valid_df = load_csv(self.data_transformation_artifact.valid_file_path)
        return train_df, valid_df

    @typechecked
    def _create_dataloader(self, dataset: Dataset, batch_size: int):
        return DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=2
        )

    @typechecked
    def train_epoch(self):
        self.model.train()
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=self.n_train_batches,
            desc=f"Train Loop [{self.i_epoch + 1}/{self.config.n_epochs}]",
        )

        for step, inputs in pbar:
            loss = self.model.forward_step(inputs)
            loss = loss / self.config.gradient_accumulation_steps
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            loss.backward()
            if (
                step + 1
            ) % self.config.gradient_accumulation_steps == 0 or step + 1 == len(
                self.train_dataloader
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

    @typechecked
    def valid_epoch(self):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(
                self.valid_dataloader,
                total=self.n_valid_batches,
                desc=f"Valid Loop [{self.i_epoch + 1}/{self.config.n_epochs}]",
            )
            for inputs in pbar:
                loss = self.model.forward_step(inputs)
                loss = loss / self.config.gradient_accumulation_steps
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    def train(self):
        for epoch in range(self.config.n_epochs):
            self.i_epoch = epoch
            self.train_epoch()
            if self.valid_dataloader:
                self.valid_epoch()

    def __call__(
        self,
    ) -> ModelTrainingArtifact:
        self.train()
        return ModelTrainingArtifact(
            model_name=self.config.name,
            model_path=self.config.outdir,
        )
