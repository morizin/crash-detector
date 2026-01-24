import torch
from .model import Model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .optimizer import get_optimizer
from ...config.config_entity import ModelTrainingConfig
from ...config.artifact_entity import ModelTrainingArtifact, ClassificationArtifact
from typeguard import typechecked
from .dataset import CachedVideoCrashDataset
from ...config.artifact_entity import DataTransformationArtifact
from ...utils.common import load_csv
from torch.utils.tensorboard import SummaryWriter
from .utils import AverageMeter
from .loss import get_loss_function
from .metrics import get_metrics
from ... import logger


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
        logger.info(f"Using device: {self.device}")

        if self.device.type == "cuda":
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")

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

        self.criterion = get_loss_function(self.config.loss)
        self.metrics = {
            metric: [get_metrics(metric), AverageMeter(), AverageMeter()]
            for metric in self.config.metrics
        }

        self.train_loss_meter = AverageMeter()
        self.valid_loss_meter = AverageMeter()

        self.writer = SummaryWriter(log_dir=str(self.config.outdir / "runs"))

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
            loss, logits = self.model.forward_step(inputs, logits=True)
            self.writer.add_scalar(
                "Train/Loss", loss.item(), self.i_epoch * self.n_train_batches + step
            )

            for metric_name, (metric_fn, train_meter, _) in self.metrics.items():
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                targets = inputs[1].cpu().numpy()
                metric_value = metric_fn(targets, preds)
                train_meter.update(metric_value, n=inputs[0].size(0))
                self.writer.add_scalar(
                    f"Train/{metric_name}",
                    metric_value,
                    self.i_epoch * self.n_train_batches + step,
                )

            loss = loss / self.config.gradient_accumulation_steps
            self.train_loss_meter.update(loss.item(), n=inputs[0].size(0))
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", avg_loss=str(self.train_loss_meter)
            )

            loss.backward()
            if (
                step + 1
            ) % self.config.gradient_accumulation_steps == 0 or step + 1 == len(
                self.train_dataloader
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

        train_loss_avg = self.train_loss_meter.avg
        self.train_loss_meter.reset()
        return train_loss_avg

    @typechecked
    def valid_epoch(self):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(
                enumerate(self.valid_dataloader),
                total=self.n_valid_batches,
                desc=f"Valid Loop [{self.i_epoch + 1}/{self.config.n_epochs}]",
            )
            for step, inputs in pbar:
                loss, logits = self.model.forward_step(inputs, logits=True)
                self.writer.add_scalar(
                    "Valid/Loss",
                    loss.item(),
                    self.i_epoch * self.n_valid_batches + step,
                )

                for metric_name, (metric_fn, _, valid_meter) in self.metrics.items():
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                    targets = inputs[1].cpu().numpy()
                    metric_value = metric_fn(targets, preds)
                    valid_meter.update(metric_value, n=inputs[0].size(0))
                    self.writer.add_scalar(
                        f"Valid/{metric_name}",
                        metric_value,
                        self.i_epoch * self.n_valid_batches + step,
                    )

                loss = loss / self.config.gradient_accumulation_steps
                self.valid_loss_meter.update(loss.item(), n=inputs[0].size(0))
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", avg_loss=str(self.valid_loss_meter)
                )

        valid_loss_avg = self.valid_loss_meter.avg
        self.valid_loss_meter.reset()
        return valid_loss_avg

    def train(self):
        for epoch in range(self.config.n_epochs):
            self.i_epoch = epoch
            train_loss = self.train_epoch()
            logger.info(
                f"Epoch {epoch + 1}/{self.config.n_epochs}, Train Loss: {train_loss:.4f}"
            )
            if self.valid_dataloader:
                valid_loss = self.valid_epoch()
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.n_epochs}, Valid Loss: {valid_loss:.4f}"
                )

        return train_loss, valid_loss if self.valid_dataloader else None

    @typechecked
    def save_model(self):
        model_path = self.config.outdir / f"{self.config.name}_model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved at: {model_path}")

    def __call__(
        self,
    ) -> ModelTrainingArtifact:
        train_loss, valid_loss = self.train()
        self.save_model()

        return ModelTrainingArtifact(
            model_name=self.config.name,
            model_path=self.config.outdir / f"{self.config.name}_model.pth",
            train_loss=train_loss,
            valid_loss=valid_loss,
            classification_artifact=ClassificationArtifact(
                **{metric: meter.avg for metric, (_, _, meter) in self.metrics.items()}
            ),
        )
