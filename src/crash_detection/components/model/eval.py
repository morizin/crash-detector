from ...config.config_entity import ModelEvaluationConfig
from .dataset import CachedVideoCrashDataset
from .model import Model
from ...utils.common import load_csv
from ...config.artifact_entity import ClassificationArtifact, ModelEvaluationArtifact
from torch.utils.data import DataLoader
from typeguard import typechecked
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from tqdm import tqdm
from ... import logger


class ModelEvaluationComponent:
    @typechecked
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        self.splits = ["valid"]
        self.valid_dataloader = self._create_dataloader(
            self.config.valid_file_path, batch_size=32
        )
        if self.config.test_file_path:
            self.splits.append("test")
            self.test_dataloader = self._create_dataloader(
                self.config.test_file_path, batch_size=32
            )

    @typechecked
    def _create_dataloader(self, filepath, batch_size: int) -> DataLoader:
        return DataLoader(
            CachedVideoCrashDataset(load_csv(filepath)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    @typechecked
    def __call__(self) -> ModelEvaluationArtifact:
        return self.evaluate()

    @typechecked
    def evaluate(self) -> ModelEvaluationArtifact:
        # Load the model
        model = Model.from_pretrained(self.config.model_path)
        model.eval()

        results = {}
        for split in self.splits:
            logger.info(f"Evaluating on {split} set...")
            results[split] = self._evaluate_split(
                model, getattr(self, f"{split}_dataloader")
            )

        return ModelEvaluationArtifact(
            name=self.config.name,
            valid_classification_artifact=results["valid"],
            test_classification_artifact=results["test"],
        )

    @typechecked
    def _evaluate_split(self, model, dataloader) -> ClassificationArtifact:
        all_preds = []
        all_labels = []
        result = {}
        with torch.no_grad():
            for input in tqdm(dataloader):
                loss, logits = model.forward_step(input, get_logits=True)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                labels = input[1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        for metric in self.config.metrics:
            if metric == "accuracy":
                acc = accuracy_score(all_labels, all_preds)
                result["accuracy"] = acc
            elif metric == "precision":
                prec = precision_score(all_labels, all_preds)
                result["precision"] = prec
            elif metric == "recall":
                rec = recall_score(all_labels, all_preds)
                result["recall"] = rec
            elif metric.replace("-", "_") == "f1_score":
                f1 = f1_score(all_labels, all_preds)
                result["f1_score"] = f1
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        return ClassificationArtifact(**result)
