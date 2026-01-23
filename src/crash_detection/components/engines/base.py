import torch
import torch.nn as nn
import torch.optim as optim
from typeguard import typechecked
from torch.utils.data import Dataset, DataLoader
from ..loss import get_loss_function
from box import ConfigDict
from tqdm import tqdm


class ClassificationEngine:
    @typechecked
    def __init__(
        self,
        config: ConfigDict,
        model: nn.Module,
        train_dataset: Dataset,
        valid_dataset: Dataset = None,
        train_batch_size: int = 32,
        valid_batch_size: int = 32,
        n_epochs: int = 10,
    ):
        self.config = config

        self.model = model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if self.device.type == "cuda":
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

        self.model.to(self.device)

        self.train_batch_size = train_batch_size
        self.train_dataloader = self._create_dataloader(
            train_dataset, self.train_batch_size
        )
        self.n_train_batches = len(self.train_dataloader)

        if valid_dataset:
            self.valid_batch_size = valid_batch_size
            self.valid_dataloader = self._create_dataloader(
                valid_dataset, self.valid_batch_size
            )
            self.n_valid_batches = len(self.valid_dataloader)
        else:
            self.valid_dataloader = self.n_valid_batches = self.valid_batch_size = None

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = get_loss_function()
        self.i_epoch = 0
        self.n_epochs = n_epochs

    @typechecked
    def _create_dataloader(self, dataset: Dataset, batch_size: int):
        return DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=2
        )

    @typechecked
    def forward_step(
        self, inputs: tuple[torch.Tensor, torch.Tensor | None]
    ) -> torch.Tensor:
        images, labels = inputs

        images = (
            images.to(self.device, non_blocking=True).float().permute(0, 3, 1, 2)
            / 255.0
        )

        logits = self.model(images)

        if labels is not None:
            labels = labels.to(self.device, non_blocking=True).float().unsqueeze(1)
            loss = self.criterion(logits, labels)
            return loss
        else:
            return logits

    @typechecked
    def train_epoch(self):
        self.model.train()
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Train Loop [{self.i_epoch + 1}/{self.n_epochs}]",
        )

        for inputs in pbar:
            self.optimizer.zero_grad()
            loss = self.forward_step(inputs)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            loss.backward()
            self.optimizer.step()

    @typechecked
    def valid_epoch(self):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(
                self.valid_dataloader,
                desc=f"Valid Loop [{self.i_epoch + 1}/{self.n_epochs}]",
            )
            for inputs in pbar:
                loss = self.forward_step(inputs)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    def train(self):
        for epoch in range(self.n_epochs):
            self.i_epoch = epoch
            self.train_epoch()
            if self.valid_dataloader:
                self.valid_epoch()
