from __future__ import annotations

"""
Full CPU-oriented experiment runner for:

Deep Reinforcement Learning for Adaptive Sequential Region Selection
in Multi-label Chest X-ray Analysis

IMPORTANT
---------
This runner is designed for the current machine:
- Intel Core i3-12100F
- 16 GB RAM
- NVIDIA GT 710 (not used)
- CPU execution

It implements:
1. DenseNet121 multi-label baseline training
2. Frozen DenseNet121 region-feature caching
3. Sequential 3x3 region-selection environment
4. PPO, DQN, and discrete SAC agents
5. Early/Mid/Final training and testing phases
6. Seeds 5, 25, and 125
7. CSV logs, checkpoints, JSON summaries, and PDF figures
8. Automatic resume after interruption or power loss

The code is intentionally conservative with memory. Full execution on CPU can
take many days. Run the benchmark command first.
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Ensar\Desktop\CXR8\cxr_drl")
SPLIT_DIR = PROJECT_DIR / "data_splits"
OUTPUT_DIR = PROJECT_DIR / "outputs_fast_48h"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "feature_cache"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
STATE_DIR = OUTPUT_DIR / "state"
CHECKPOINT_INTERVAL_EPISODES = 50

TRAIN_CSV = SPLIT_DIR / "train.csv"
VALIDATION_CSV = SPLIT_DIR / "validation.csv"
TEST_CSV = SPLIT_DIR / "test.csv"


# ============================================================
# FIXED METHODOLOGY CONFIGURATION
# ============================================================

DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

SEEDS = [5, 25, 125]

PHASES = {
    "early": {"train_episodes": 1000, "test_episodes": 2000},
    "mid": {"train_episodes": 2000, "test_episodes": 3000},
    "final": {"train_episodes": 3000, "test_episodes": 4000},
}

ACTIONS = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
    4: "Stop",
}

NUM_LABELS = 14
NUM_ACTIONS = 5
GRID_SIZE = 3
NUM_REGIONS = GRID_SIZE * GRID_SIZE
IMAGE_SIZE = 224
REGION_SIZE = 224
FEATURE_DIM = 1024
MAX_REGION_STEPS = 10

GAMMA = 0.99
LEARNING_RATE = 3e-4

# CPU-safe settings.
BASELINE_BATCH_SIZE = 8
FEATURE_BATCH_SIZE = 8
NUM_WORKERS = 0
BASELINE_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 2

# Reward weights.
REWARD_IMPROVEMENT_WEIGHT = 1.0
REVISIT_PENALTY = 0.05
STEP_PENALTY = 0.01
STOP_BONUS = 0.10
INVALID_MOVE_PENALTY = 0.05

# Evaluation threshold for multi-label outputs.
CLASSIFICATION_THRESHOLD = 0.5


@dataclass
class Config:
    project_dir: str = str(PROJECT_DIR)
    dataset: str = "NIH ChestX-ray14"
    image_size: int = IMAGE_SIZE
    region_size: int = REGION_SIZE
    num_labels: int = NUM_LABELS
    grid_size: int = GRID_SIZE
    num_actions: int = NUM_ACTIONS
    max_region_steps: int = MAX_REGION_STEPS
    gamma: float = GAMMA
    learning_rate: float = LEARNING_RATE
    baseline_batch_size: int = BASELINE_BATCH_SIZE
    feature_batch_size: int = FEATURE_BATCH_SIZE
    baseline_epochs: int = BASELINE_EPOCHS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    seeds: Tuple[int, ...] = tuple(SEEDS)
    phases: Dict[str, Dict[str, int]] = None

    def __post_init__(self) -> None:
        if self.phases is None:
            self.phases = PHASES


# ============================================================
# REPRODUCIBILITY AND UTILITIES
# ============================================================

def ensure_directories() -> None:
    for directory in [
        OUTPUT_DIR,
        CHECKPOINT_DIR,
        LOG_DIR,
        FIGURE_DIR,
        CACHE_DIR,
        SUMMARY_DIR,
        STATE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    # Force CPU on this machine. GT 710 is intentionally not used.
    return torch.device("cpu")


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(payload: object, path: Path) -> None:
    """Write a PyTorch checkpoint atomically to survive power loss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def marker_path(name: str) -> Path:
    return STATE_DIR / f"{name}.done.json"


def mark_done(name: str, details: Optional[Dict[str, object]] = None) -> None:
    save_json(
        marker_path(name),
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "details": details or {},
        },
    )


def is_done(name: str) -> bool:
    return marker_path(name).is_file()


def last_logged_integer(path: Path, column: str) -> int:
    if not path.is_file():
        return 0
    try:
        frame = pd.read_csv(path, usecols=[column])
        if frame.empty:
            return 0
        return int(frame[column].max())
    except Exception:
        return 0


def append_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def verify_required_files() -> None:
    missing = [
        str(path)
        for path in [TRAIN_CSV, VALIDATION_CSV, TEST_CSV]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required split files:\n" + "\n".join(missing)
        )


# ============================================================
# DATASET
# ============================================================

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path: Path, training: bool = False) -> None:
        self.csv_path = Path(csv_path)
        self.training = training
        self.dataframe = pd.read_csv(self.csv_path)

        required = {"Image Index", "Patient ID", "image_path", *DISEASE_LABELS}
        missing = required.difference(self.dataframe.columns)
        if missing:
            raise ValueError(
                f"{self.csv_path.name} missing columns: {sorted(missing)}"
            )

        common = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ]

        if training:
            common.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(5),
                ]
            )

        common.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.transform = transforms.Compose(common)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.dataframe.iloc[index]
        path = Path(str(row["image_path"]))

        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        labels = torch.tensor(
            row[DISEASE_LABELS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        return {
            "image": tensor,
            "labels": labels,
            "image_index": str(row["Image Index"]),
            "patient_id": int(row["Patient ID"]),
            "image_path": str(path),
            "dataset_index": int(index),
        }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        generator=generator,
    )


def positive_class_weights(train_csv: Path) -> torch.Tensor:
    frame = pd.read_csv(train_csv, usecols=DISEASE_LABELS)
    positives = frame[DISEASE_LABELS].sum(axis=0).to_numpy(dtype=np.float64)
    negatives = len(frame) - positives
    weights = negatives / np.maximum(positives, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# DENSENET121 BASELINE
# ============================================================

class DenseNetMultiLabel(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        weights = (
            models.DenseNet121_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        self.backbone = models.densenet121(weights=weights)
        input_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(input_features, NUM_LABELS)

        # CPU-fast protocol: freeze the convolutional backbone and train only
        # the 14-label classification head. This must be stated in the paper.
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.features(x)
            features = F.relu(features, inplace=False)
            pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.backbone.classifier(pooled)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)
        features = F.relu(features, inplace=False)
        return F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

    def extract_global_and_regions(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # One DenseNet pass gives both global and 3x3 regional embeddings.
        features = self.backbone.features(x)
        features = F.relu(features, inplace=False)
        global_features = F.adaptive_avg_pool2d(
            features, (1, 1)
        ).flatten(1)
        region_features = F.adaptive_avg_pool2d(
            features, (GRID_SIZE, GRID_SIZE)
        ).flatten(2).transpose(1, 2)
        return global_features, region_features


def multilabel_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> Dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int32)

    metrics: Dict[str, float] = {
        "micro_f1": float(
            f1_score(labels, predictions, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "micro_precision": float(
            precision_score(
                labels, predictions, average="micro", zero_division=0
            )
        ),
        "micro_recall": float(
            recall_score(
                labels, predictions, average="micro", zero_division=0
            )
        ),
    }

    try:
        metrics["macro_auc"] = float(
            roc_auc_score(labels, probabilities, average="macro")
        )
    except ValueError:
        metrics["macro_auc"] = float("nan")

    try:
        metrics["micro_auc"] = float(
            roc_auc_score(labels, probabilities, average="micro")
        )
    except ValueError:
        metrics["micro_auc"] = float("nan")

    try:
        metrics["mean_average_precision"] = float(
            average_precision_score(labels, probabilities, average="macro")
        )
    except ValueError:
        metrics["mean_average_precision"] = float("nan")

    return metrics


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    labels_all: List[np.ndarray] = []
    probabilities_all: List[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(device())
        labels = batch["labels"].to(device())

        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits)

        losses.append(float(loss.item()))
        labels_all.append(labels.cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())

    labels_np = np.concatenate(labels_all, axis=0)
    probabilities_np = np.concatenate(probabilities_all, axis=0)

    metrics = multilabel_metrics(labels_np, probabilities_np)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_baseline(
    seed: int,
    max_train_images: Optional[int] = None,
    max_validation_images: Optional[int] = None,
) -> Path:
    set_seed(seed)
    print(f"\n[Baseline] Seed {seed}")

    completion_name = f"baseline_seed{seed}"
    best_path = CHECKPOINT_DIR / f"densenet121_seed{seed}_best.pt"
    latest_path = CHECKPOINT_DIR / f"densenet121_seed{seed}_latest.pt"
    log_path = LOG_DIR / f"densenet121_seed{seed}_training.csv"

    if is_done(completion_name) and best_path.exists():
        print(f"  Already complete. Using {best_path}")
        return best_path

    train_dataset: Dataset = ChestXrayDataset(TRAIN_CSV, training=True)
    validation_dataset: Dataset = ChestXrayDataset(
        VALIDATION_CSV, training=False
    )

    if max_train_images is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(max_train_images, len(train_dataset))),
        )
    if max_validation_images is not None:
        validation_dataset = Subset(
            validation_dataset,
            range(min(max_validation_images, len(validation_dataset))),
        )

    train_loader = make_loader(
        train_dataset, BASELINE_BATCH_SIZE, True, seed
    )
    validation_loader = make_loader(
        validation_dataset, BASELINE_BATCH_SIZE, False, seed
    )

    model = DenseNetMultiLabel(pretrained=not latest_path.exists()).to(device())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_class_weights(TRAIN_CSV).to(device())
    )
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 1
    best_auc = -math.inf
    epochs_without_improvement = 0

    if latest_path.exists():
        checkpoint = torch.load(
            latest_path, map_location=device(), weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_auc = float(checkpoint.get("best_auc", -math.inf))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        print(f"  Resuming from epoch {start_epoch}")

    if start_epoch > BASELINE_EPOCHS:
        if best_path.exists():
            mark_done(completion_name, {"best_auc": best_auc})
            return best_path
        raise RuntimeError("Latest baseline exists but best checkpoint is missing.")

    for epoch in range(start_epoch, BASELINE_EPOCHS + 1):
        model.train()
        epoch_losses: List[float] = []
        start_time = time.perf_counter()

        for batch_index, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device())
            labels = batch["labels"].to(device())

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

            if batch_index % 100 == 0:
                print(
                    f"  epoch={epoch} batch={batch_index}/"
                    f"{len(train_loader)} loss={np.mean(epoch_losses[-100:]):.4f}"
                )

        validation_metrics = evaluate_classifier(
            model, validation_loader, criterion
        )
        duration = time.perf_counter() - start_time

        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": validation_metrics["loss"],
            "validation_macro_auc": validation_metrics["macro_auc"],
            "validation_micro_auc": validation_metrics["micro_auc"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_micro_f1": validation_metrics["micro_f1"],
            "epoch_seconds": duration,
        }
        append_csv(log_path, row)
        print("  ", row)

        score = validation_metrics["macro_auc"]
        if not math.isnan(score) and score > best_auc:
            best_auc = score
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": validation_metrics,
                    "config": asdict(Config()),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        atomic_torch_save(
            {
                "seed": seed,
                "epoch": epoch,
                "best_auc": best_auc,
                "epochs_without_improvement": epochs_without_improvement,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_metrics": validation_metrics,
                "config": asdict(Config()),
            },
            latest_path,
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("  Early stopping activated.")
            break

    if not best_path.exists():
        raise RuntimeError("Baseline best checkpoint was not created.")

    mark_done(completion_name, {"best_auc": best_auc})
    return best_path


# ============================================================
# REGION FEATURE CACHE
# ============================================================

def crop_regions(images: torch.Tensor) -> torch.Tensor:
    """
    Return a 3x3 grid of resized crops:
    input:  [B, 3, 224, 224]
    output: [B, 9, 3, 224, 224]
    """
    batch_size, _, height, width = images.shape
    h_step = height // GRID_SIZE
    w_step = width // GRID_SIZE

    regions: List[torch.Tensor] = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y0 = row * h_step
            x0 = col * w_step
            y1 = height if row == GRID_SIZE - 1 else (row + 1) * h_step
            x1 = width if col == GRID_SIZE - 1 else (col + 1) * w_step

            crop = images[:, :, y0:y1, x0:x1]
            crop = F.interpolate(
                crop,
                size=(REGION_SIZE, REGION_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            regions.append(crop)

    return torch.stack(regions, dim=1)


def load_trained_baseline(checkpoint_path: Path) -> DenseNetMultiLabel:
    model = DenseNetMultiLabel(pretrained=False).to(device())
    checkpoint = torch.load(
        checkpoint_path, map_location=device(), weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def cache_features_for_split(
    split_name: str,
    csv_path: Path,
    baseline_checkpoint: Path,
    seed: int,
    max_images: Optional[int] = None,
) -> Path:
    cache_path = CACHE_DIR / f"{split_name}_features_shared.pt"
    completion_name = f"cache_{split_name}_shared"
    if cache_path.exists() and is_done(completion_name):
        print(f"[Cache] Existing complete cache used: {cache_path}")
        return cache_path
    if cache_path.exists() and not is_done(completion_name):
        print(f"[Cache] Removing incomplete cache: {cache_path}")
        cache_path.unlink()

    dataset: Dataset = ChestXrayDataset(csv_path, training=False)
    if max_images is not None:
        dataset = Subset(dataset, range(min(max_images, len(dataset))))

    loader = make_loader(dataset, FEATURE_BATCH_SIZE, False, seed)
    model = load_trained_baseline(baseline_checkpoint)

    all_global: List[torch.Tensor] = []
    all_regions: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_indices: List[str] = []
    all_patients: List[int] = []

    print(f"[Cache] {split_name}: {len(dataset):,} images")

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device())
        labels = batch["labels"]

        # One backbone forward pass instead of one global + nine crop passes.
        global_features, region_features = (
            model.extract_global_and_regions(images)
        )
        global_features = global_features.cpu().half()
        region_features = region_features.cpu().half()

        all_global.append(global_features)
        all_regions.append(region_features)
        all_labels.append(labels.cpu())
        all_indices.extend(list(batch["image_index"]))
        all_patients.extend([int(x) for x in batch["patient_id"]])

        if batch_index % 50 == 0:
            print(f"  cached {batch_index * FEATURE_BATCH_SIZE:,}")

    payload = {
        "global_features": torch.cat(all_global, dim=0),
        "region_features": torch.cat(all_regions, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "image_indices": all_indices,
        "patient_ids": all_patients,
        "split": split_name,
        "seed": "shared_frozen_backbone",
    }
    atomic_torch_save(payload, cache_path)
    mark_done(
        completion_name,
        {"split": split_name, "seed": "shared", "images": len(dataset)},
    )
    return cache_path


# ============================================================
# SEQUENTIAL REGION-SELECTION ENVIRONMENT
# ============================================================

class CachedRegionDataset:
    def __init__(self, cache_path: Path) -> None:
        payload = torch.load(
            cache_path, map_location="cpu", weights_only=False
        )
        self.global_features = payload["global_features"].float()
        self.region_features = payload["region_features"].float()
        self.labels = payload["labels"].float()
        self.image_indices = payload["image_indices"]
        self.patient_ids = payload["patient_ids"]

    def __len__(self) -> int:
        return self.labels.shape[0]


class RegionClassifier(nn.Module):
    """
    Predicts 14 labels from an aggregated feature representation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, NUM_LABELS),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature)


class RegionSelectionEnv:
    def __init__(
        self,
        dataset: CachedRegionDataset,
        classifier: RegionClassifier,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.classifier = classifier
        self.rng = np.random.default_rng(seed)

        self.index = 0
        self.position = 4
        self.steps = 0
        self.visited: set[int] = set()
        self.aggregated_feature = torch.zeros(FEATURE_DIM)
        self.previous_loss = 0.0
        self.done = False

    @property
    def state_dim(self) -> int:
        # global feature + current region feature + normalized location (2)
        # + visit mask (9) + remaining budget (1)
        return FEATURE_DIM * 2 + 2 + NUM_REGIONS + 1

    def _position_coordinates(self, position: int) -> Tuple[int, int]:
        return position // GRID_SIZE, position % GRID_SIZE

    def _state(self) -> torch.Tensor:
        global_feature = self.dataset.global_features[self.index]
        current_feature = self.dataset.region_features[
            self.index, self.position
        ]

        row, col = self._position_coordinates(self.position)
        location = torch.tensor(
            [
                row / (GRID_SIZE - 1),
                col / (GRID_SIZE - 1),
            ],
            dtype=torch.float32,
        )

        visit_mask = torch.zeros(NUM_REGIONS, dtype=torch.float32)
        for item in self.visited:
            visit_mask[item] = 1.0

        remaining = torch.tensor(
            [1.0 - self.steps / MAX_REGION_STEPS],
            dtype=torch.float32,
        )

        return torch.cat(
            [
                global_feature,
                current_feature,
                location,
                visit_mask,
                remaining,
            ]
        )

    def _classification_loss(self) -> float:
        label = self.dataset.labels[self.index].unsqueeze(0)
        logits = self.classifier(
            self.aggregated_feature.unsqueeze(0)
        )
        loss = F.binary_cross_entropy_with_logits(logits, label)
        return float(loss.item())

    def reset(self, index: Optional[int] = None) -> torch.Tensor:
        self.index = (
            int(index)
            if index is not None
            else int(self.rng.integers(0, len(self.dataset)))
        )
        self.position = 4
        self.steps = 0
        self.visited = {self.position}
        self.aggregated_feature = self.dataset.region_features[
            self.index, self.position
        ].clone()
        self.previous_loss = self._classification_loss()
        self.done = False
        return self._state()

    def _move(self, action: int) -> Tuple[int, bool]:
        row, col = self._position_coordinates(self.position)
        new_row, new_col = row, col

        if action == 0:
            new_row -= 1
        elif action == 1:
            new_row += 1
        elif action == 2:
            new_col -= 1
        elif action == 3:
            new_col += 1

        valid = (
            0 <= new_row < GRID_SIZE
            and 0 <= new_col < GRID_SIZE
        )
        if not valid:
            return self.position, False

        return new_row * GRID_SIZE + new_col, True

    def step(
        self, action: int
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, object]]:
        if self.done:
            raise RuntimeError("step() called after episode termination.")

        self.steps += 1
        reward = -STEP_PENALTY
        revisit = False
        invalid = False

        if action == 4:
            self.done = True
            reward += STOP_BONUS
        else:
            new_position, valid = self._move(action)
            if not valid:
                invalid = True
                reward -= INVALID_MOVE_PENALTY
            else:
                self.position = new_position
                revisit = self.position in self.visited

                if revisit:
                    reward -= REVISIT_PENALTY
                else:
                    self.visited.add(self.position)

                selected = self.dataset.region_features[
                    self.index, self.position
                ]
                count = float(len(self.visited))
                self.aggregated_feature = (
                    self.aggregated_feature * (count - 1.0) + selected
                ) / count

                current_loss = self._classification_loss()
                improvement = self.previous_loss - current_loss
                reward += REWARD_IMPROVEMENT_WEIGHT * improvement
                self.previous_loss = current_loss

        if self.steps >= MAX_REGION_STEPS:
            self.done = True

        info = {
            "image_index": self.dataset.image_indices[self.index],
            "patient_id": self.dataset.patient_ids[self.index],
            "steps": self.steps,
            "selected_regions": len(self.visited),
            "revisit": revisit,
            "invalid_move": invalid,
            "classification_loss": self.previous_loss,
            "stopped": action == 4,
        }

        return self._state(), float(reward), self.done, info


# ============================================================
# SHARED NETWORKS
# ============================================================

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# DQN
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                state.cpu(),
                int(action),
                float(reward),
                next_state.cpu(),
                float(done),
            )
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.buffer.maxlen,
            "items": list(self.buffer),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        capacity = int(state.get("capacity", 50_000))
        self.buffer = deque(state.get("items", []), maxlen=capacity)


class DQNAgent:
    def __init__(self, state_dim: int, seed: int) -> None:
        set_seed(seed)
        self.q = MLP(state_dim, NUM_ACTIONS).to(device())
        self.target = MLP(state_dim, NUM_ACTIONS).to(device())
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = Adam(self.q.parameters(), lr=LEARNING_RATE)
        self.replay = ReplayBuffer()
        self.batch_size = 64
        self.target_update = 200
        self.updates = 0

    def act(
        self,
        state: torch.Tensor,
        training: bool,
        episode: int = 0,
    ) -> int:
        epsilon = (
            max(0.05, 1.0 - episode / 2000.0)
            if training
            else 0.0
        )
        if training and random.random() < epsilon:
            return random.randrange(NUM_ACTIONS)

        with torch.no_grad():
            q_values = self.q(state.unsqueeze(0).to(device()))
        return int(q_values.argmax(dim=1).item())

    def observe(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> Optional[float]:
        self.replay.push(state, action, reward, next_state, done)
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )
        states = states.to(device())
        actions = actions.to(device())
        rewards = rewards.to(device())
        next_states = next_states.to(device())
        dones = dones.to(device())

        q_values = self.q(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            targets = rewards + GAMMA * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def checkpoint_state(self) -> Dict[str, object]:
        return {
            "type": "DQN",
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "updates": self.updates,
        }

    def load_checkpoint_state(self, state: Dict[str, object]) -> None:
        self.q.load_state_dict(state["q"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state.get("replay", {}))
        self.updates = int(state.get("updates", 0))

    def save(self, path: Path) -> None:
        atomic_torch_save(self.checkpoint_state(), path)


# ============================================================
# PPO
# ============================================================

class PPOAgent:
    def __init__(self, state_dim: int, seed: int) -> None:
        set_seed(seed)
        self.actor = MLP(state_dim, NUM_ACTIONS).to(device())
        self.critic = MLP(state_dim, 1).to(device())
        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=LEARNING_RATE,
        )
        self.clip_ratio = 0.2
        self.update_epochs = 4

    def act(
        self,
        state: torch.Tensor,
        training: bool,
        episode: int = 0,
    ) -> Tuple[int, float, float]:
        with torch.no_grad():
            logits = self.actor(state.unsqueeze(0).to(device()))
            logits = torch.nan_to_num(
                logits, nan=0.0, posinf=20.0, neginf=-20.0
            )
            distribution = Categorical(logits=logits)
            action = (
                distribution.sample()
                if training
                else logits.argmax(dim=1)
            )
            log_probability = distribution.log_prob(action)
            value = self.critic(
                state.unsqueeze(0).to(device())
            ).squeeze(1)

        return (
            int(action.item()),
            float(log_probability.item()),
            float(value.item()),
        )

    def update(self, trajectory: List[Dict[str, object]]) -> float:
        states = torch.stack(
            [item["state"] for item in trajectory]
        ).to(device())
        actions = torch.tensor(
            [item["action"] for item in trajectory],
            dtype=torch.long,
            device=device(),
        )
        old_log_probs = torch.tensor(
            [item["log_prob"] for item in trajectory],
            dtype=torch.float32,
            device=device(),
        )
        rewards = [float(item["reward"]) for item in trajectory]
        dones = [bool(item["done"]) for item in trajectory]

        returns: List[float] = []
        running = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running = 0.0
            running = reward + GAMMA * running
            returns.append(running)
        returns.reverse()

        returns_tensor = torch.tensor(
            returns, dtype=torch.float32, device=device()
        )

        with torch.no_grad():
            values = self.critic(states).squeeze(1)
            advantages = returns_tensor - values
            # torch.std() is unbiased by default. For a one-step episode,
            # that produces NaN because the degrees of freedom are zero.
            # Short episodes are valid here because the agent may select Stop
            # immediately, so normalize safely.
            if advantages.numel() > 1:
                advantage_std = advantages.std(unbiased=False)
                advantages = (
                    advantages - advantages.mean()
                ) / advantage_std.clamp_min(1e-8)
            else:
                advantages = advantages - advantages.mean()
            advantages = torch.nan_to_num(
                advantages, nan=0.0, posinf=0.0, neginf=0.0
            )

        losses: List[float] = []

        for _ in range(self.update_epochs):
            logits = self.actor(states)
            logits = torch.nan_to_num(
                logits, nan=0.0, posinf=20.0, neginf=-20.0
            )
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratios * advantages
            clipped = torch.clamp(
                ratios,
                1.0 - self.clip_ratio,
                1.0 + self.clip_ratio,
            ) * advantages

            actor_loss = -torch.min(unclipped, clipped).mean()
            predicted_values = self.critic(states).squeeze(1)
            critic_loss = F.mse_loss(
                predicted_values, returns_tensor
            )

            total_loss = (
                actor_loss + 0.5 * critic_loss - 0.01 * entropy
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    "Non-finite PPO loss detected. The current episode "
                    "was not applied to the model."
                )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters())
                + list(self.critic.parameters()),
                1.0,
            )
            self.optimizer.step()
            losses.append(float(total_loss.item()))

        return float(np.mean(losses))

    def checkpoint_state(self) -> Dict[str, object]:
        return {
            "type": "PPO",
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_checkpoint_state(self, state: Dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])

    def save(self, path: Path) -> None:
        atomic_torch_save(self.checkpoint_state(), path)


# ============================================================
# DISCRETE SAC
# ============================================================

class DiscreteSACAgent:
    def __init__(self, state_dim: int, seed: int) -> None:
        set_seed(seed)
        self.actor = MLP(state_dim, NUM_ACTIONS).to(device())
        self.q1 = MLP(state_dim, NUM_ACTIONS).to(device())
        self.q2 = MLP(state_dim, NUM_ACTIONS).to(device())
        self.target_q1 = MLP(state_dim, NUM_ACTIONS).to(device())
        self.target_q2 = MLP(state_dim, NUM_ACTIONS).to(device())

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=LEARNING_RATE
        )
        self.q_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=LEARNING_RATE,
        )

        self.replay = ReplayBuffer()
        self.batch_size = 64
        self.alpha = 0.2
        self.tau = 0.005

    def act(
        self,
        state: torch.Tensor,
        training: bool,
        episode: int = 0,
    ) -> int:
        with torch.no_grad():
            logits = self.actor(state.unsqueeze(0).to(device()))
            probabilities = F.softmax(logits, dim=1)
            if training:
                action = Categorical(probabilities).sample()
            else:
                action = probabilities.argmax(dim=1)
        return int(action.item())

    def observe(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> Optional[float]:
        self.replay.push(state, action, reward, next_state, done)
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )
        states = states.to(device())
        actions = actions.to(device())
        rewards = rewards.to(device())
        next_states = next_states.to(device())
        dones = dones.to(device())

        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_log_probs = F.log_softmax(next_logits, dim=1)
            next_probs = next_log_probs.exp()

            target_q = torch.min(
                self.target_q1(next_states),
                self.target_q2(next_states),
            )
            next_v = (
                next_probs * (target_q - self.alpha * next_log_probs)
            ).sum(dim=1)
            target = rewards + GAMMA * (1.0 - dones) * next_v

        q1_pred = self.q1(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        q2_pred = self.q2(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(
            q2_pred, target
        )

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        logits = self.actor(states)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        min_q = torch.min(self.q1(states), self.q2(states))
        actor_loss = (
            probs * (self.alpha * log_probs - min_q)
        ).sum(dim=1).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_q1.parameters(), self.q1.parameters()
            ):
                target_parameter.data.mul_(1.0 - self.tau)
                target_parameter.data.add_(self.tau * parameter.data)

            for target_parameter, parameter in zip(
                self.target_q2.parameters(), self.q2.parameters()
            ):
                target_parameter.data.mul_(1.0 - self.tau)
                target_parameter.data.add_(self.tau * parameter.data)

        return float((q_loss + actor_loss).item())

    def checkpoint_state(self) -> Dict[str, object]:
        return {
            "type": "DiscreteSAC",
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "replay": self.replay.state_dict(),
        }

    def load_checkpoint_state(self, state: Dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.target_q1.load_state_dict(state["target_q1"])
        self.target_q2.load_state_dict(state["target_q2"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.replay.load_state_dict(state.get("replay", {}))

    def save(self, path: Path) -> None:
        atomic_torch_save(self.checkpoint_state(), path)


# ============================================================
# REGION CLASSIFIER PRETRAINING
# ============================================================

def train_region_classifier(
    train_cache: Path,
    validation_cache: Path,
    seed: int,
    epochs: int = 5,
) -> Path:
    set_seed(seed)
    best_path = CHECKPOINT_DIR / f"region_classifier_seed{seed}_best.pt"
    latest_path = CHECKPOINT_DIR / f"region_classifier_seed{seed}_latest.pt"
    completion_name = f"region_classifier_seed{seed}"

    if is_done(completion_name) and best_path.exists():
        return best_path

    train_data = CachedRegionDataset(train_cache)
    validation_data = CachedRegionDataset(validation_cache)

    classifier = RegionClassifier().to(device())
    optimizer = Adam(classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_class_weights(TRAIN_CSV).to(device())
    )

    batch_size = 128
    best_loss = math.inf
    start_epoch = 1

    if latest_path.exists():
        payload = torch.load(
            latest_path, map_location=device(), weights_only=False
        )
        classifier.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        best_loss = float(payload.get("best_loss", math.inf))
        start_epoch = int(payload["epoch"]) + 1
        print(
            f"[Region classifier] seed={seed}, resuming at epoch "
            f"{start_epoch}"
        )

    for epoch in range(start_epoch, epochs + 1):
        classifier.train()
        indices = np.random.permutation(len(train_data))
        train_losses: List[float] = []

        for start_index in range(0, len(indices), batch_size):
            selected = indices[start_index : start_index + batch_size]
            features = train_data.region_features[selected].mean(dim=1)
            labels = train_data.labels[selected]

            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        classifier.eval()
        validation_losses: List[float] = []
        with torch.no_grad():
            for start_index in range(
                0, len(validation_data), batch_size
            ):
                features = validation_data.region_features[
                    start_index : start_index + batch_size
                ].mean(dim=1)
                labels = validation_data.labels[
                    start_index : start_index + batch_size
                ]
                validation_losses.append(
                    float(criterion(classifier(features), labels).item())
                )

        validation_loss = float(np.mean(validation_losses))
        append_csv(
            LOG_DIR / f"region_classifier_seed{seed}.csv",
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
            },
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            atomic_torch_save(classifier.state_dict(), best_path)

        atomic_torch_save(
            {
                "epoch": epoch,
                "best_loss": best_loss,
                "model": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            latest_path,
        )

    if not best_path.exists():
        raise RuntimeError("Region classifier best checkpoint is missing.")

    mark_done(completion_name, {"best_loss": best_loss})
    return best_path


def load_region_classifier(path: Path) -> RegionClassifier:
    classifier = RegionClassifier().to(device())
    classifier.load_state_dict(
        torch.load(path, map_location=device(), weights_only=True)
    )
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad = False
    return classifier


# ============================================================
# DRL TRAINING AND EVALUATION
# ============================================================

def create_agent(name: str, state_dim: int, seed: int):
    normalized = name.lower()
    if normalized == "ppo":
        return PPOAgent(state_dim, seed)
    if normalized == "dqn":
        return DQNAgent(state_dim, seed)
    if normalized in {"discretesac", "sac", "discrete_sac"}:
        return DiscreteSACAgent(state_dim, seed)
    raise ValueError(f"Unsupported agent: {name}")


def run_training_episode(
    agent_name: str,
    agent,
    env: RegionSelectionEnv,
    episode: int,
) -> Dict[str, float]:
    state = env.reset()
    total_reward = 0.0
    losses: List[float] = []
    trajectory: List[Dict[str, object]] = []
    final_info: Dict[str, object] = {}

    while True:
        if agent_name.lower() == "ppo":
            action, log_prob, value = agent.act(
                state, training=True, episode=episode
            )
        else:
            action = agent.act(
                state, training=True, episode=episode
            )
            log_prob = 0.0
            value = 0.0

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if agent_name.lower() == "ppo":
            trajectory.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "log_prob": log_prob,
                    "value": value,
                }
            )
        else:
            loss = agent.observe(
                state, action, reward, next_state, done
            )
            if loss is not None:
                losses.append(loss)

        state = next_state
        final_info = info

        if done:
            break

    if agent_name.lower() == "ppo":
        losses.append(agent.update(trajectory))

    return {
        "reward": float(total_reward),
        "steps": float(final_info["steps"]),
        "selected_regions": float(final_info["selected_regions"]),
        "classification_loss": float(
            final_info["classification_loss"]
        ),
        "optimization_loss": float(
            np.mean(losses) if losses else np.nan
        ),
    }


@torch.no_grad()
def evaluate_agent(
    agent_name: str,
    agent,
    env: RegionSelectionEnv,
    episodes: int,
    seed: int,
    phase: str,
) -> Dict[str, float]:
    rewards: List[float] = []
    steps: List[int] = []
    selected_regions: List[int] = []
    classification_losses: List[float] = []
    stop_count = 0

    rng = np.random.default_rng(seed + 10_000)
    sampled_indices = rng.integers(
        0, len(env.dataset), size=episodes
    )

    for evaluation_episode, index in enumerate(
        sampled_indices, start=1
    ):
        state = env.reset(int(index))
        total_reward = 0.0
        final_info: Dict[str, object] = {}

        while True:
            if agent_name.lower() == "ppo":
                action, _, _ = agent.act(
                    state, training=False
                )
            else:
                action = agent.act(state, training=False)

            state, reward, done, info = env.step(action)
            total_reward += reward
            final_info = info

            if done:
                break

        rewards.append(total_reward)
        steps.append(int(final_info["steps"]))
        selected_regions.append(
            int(final_info["selected_regions"])
        )
        classification_losses.append(
            float(final_info["classification_loss"])
        )
        stop_count += int(bool(final_info["stopped"]))

        append_csv(
            LOG_DIR
            / f"{agent_name}_seed{seed}_{phase}_evaluation_episodes.csv",
            {
                "agent": agent_name,
                "seed": seed,
                "phase": phase,
                "evaluation_episode": evaluation_episode,
                "reward": total_reward,
                "steps": final_info["steps"],
                "selected_regions": final_info["selected_regions"],
                "classification_loss": final_info[
                    "classification_loss"
                ],
                "stopped": final_info["stopped"],
            },
        )

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps)),
        "mean_selected_regions": float(np.mean(selected_regions)),
        "mean_classification_loss": float(
            np.mean(classification_losses)
        ),
        "stop_rate": float(stop_count / episodes),
    }


def train_agent_phases(
    agent_name: str,
    seed: int,
    train_cache: Path,
    test_cache: Path,
    classifier_checkpoint: Path,
    phase_scale: float = 1.0,
) -> None:
    set_seed(seed)

    train_dataset = CachedRegionDataset(train_cache)
    test_dataset = CachedRegionDataset(test_cache)
    classifier = load_region_classifier(classifier_checkpoint)

    train_env = RegionSelectionEnv(train_dataset, classifier, seed)
    test_env = RegionSelectionEnv(
        test_dataset, classifier, seed + 1000
    )
    agent = create_agent(agent_name, train_env.state_dim, seed)

    cumulative_episode = 0

    for phase, counts in PHASES.items():
        train_episodes = max(
            1, int(counts["train_episodes"] * phase_scale)
        )
        test_episodes = max(
            1, int(counts["test_episodes"] * phase_scale)
        )

        phase_name = f"{agent_name}_seed{seed}_{phase}"
        phase_log = LOG_DIR / f"{phase_name}_training.csv"
        latest_checkpoint = (
            CHECKPOINT_DIR / f"{phase_name}_latest.pt"
        )
        final_checkpoint = (
            CHECKPOINT_DIR / f"{phase_name}_final.pt"
        )

        if is_done(phase_name) and final_checkpoint.exists():
            print(f"\n[{phase_name}] already complete. Skipping.")
            cumulative_episode += train_episodes
            continue

        start_local_episode = last_logged_integer(
            phase_log, "local_episode"
        ) + 1

        if latest_checkpoint.exists():
            payload = torch.load(
                latest_checkpoint,
                map_location=device(),
                weights_only=False,
            )
            agent.load_checkpoint_state(payload["agent"])
            start_local_episode = max(
                start_local_episode,
                int(payload["local_episode"]) + 1,
            )
            cumulative_episode = int(
                payload.get("cumulative_episode", cumulative_episode)
            )
            random.setstate(payload["python_random_state"])
            np.random.set_state(payload["numpy_random_state"])
            torch.set_rng_state(payload["torch_random_state"])
            print(
                f"\n[{phase_name}] resuming from local episode "
                f"{start_local_episode}"
            )
        else:
            print(
                f"\n[{agent_name}] seed={seed} phase={phase} "
                f"train={train_episodes} test={test_episodes}"
            )

        for local_episode in range(
            start_local_episode, train_episodes + 1
        ):
            cumulative_episode += 1
            result = run_training_episode(
                agent_name,
                agent,
                train_env,
                cumulative_episode,
            )

            append_csv(
                phase_log,
                {
                    "agent": agent_name,
                    "seed": seed,
                    "phase": phase,
                    "local_episode": local_episode,
                    "cumulative_episode": cumulative_episode,
                    **result,
                },
            )

            if (
                local_episode % CHECKPOINT_INTERVAL_EPISODES == 0
                or local_episode == train_episodes
            ):
                atomic_torch_save(
                    {
                        "agent": agent.checkpoint_state(),
                        "agent_name": agent_name,
                        "seed": seed,
                        "phase": phase,
                        "local_episode": local_episode,
                        "cumulative_episode": cumulative_episode,
                        "python_random_state": random.getstate(),
                        "numpy_random_state": np.random.get_state(),
                        "torch_random_state": torch.get_rng_state(),
                    },
                    latest_checkpoint,
                )

            if local_episode % 100 == 0:
                print(
                    f"  episode={local_episode}/{train_episodes} "
                    f"reward={result['reward']:.4f} "
                    f"steps={result['steps']:.0f}"
                )

        agent.save(final_checkpoint)

        evaluation_marker = f"{phase_name}_evaluation"
        evaluation_summary_path = (
            SUMMARY_DIR / f"{phase_name}_evaluation.json"
        )

        if is_done(evaluation_marker) and evaluation_summary_path.exists():
            evaluation = json.loads(
                evaluation_summary_path.read_text(encoding="utf-8")
            )
        else:
            evaluation_log = (
                LOG_DIR
                / f"{agent_name}_seed{seed}_{phase}_evaluation_episodes.csv"
            )
            if evaluation_log.exists():
                evaluation_log.unlink()

            evaluation = evaluate_agent(
                agent_name,
                agent,
                test_env,
                test_episodes,
                seed,
                phase,
            )
            save_json(evaluation_summary_path, evaluation)
            mark_done(evaluation_marker, evaluation)

        summary_file = SUMMARY_DIR / "drl_phase_summary.csv"
        existing = pd.DataFrame()
        if summary_file.exists():
            existing = pd.read_csv(summary_file)

        duplicate = False
        if not existing.empty:
            duplicate = bool(
                (
                    (existing["agent"] == agent_name)
                    & (existing["seed"] == seed)
                    & (existing["phase"] == phase)
                ).any()
            )

        if not duplicate:
            append_csv(
                summary_file,
                {
                    "agent": agent_name,
                    "seed": seed,
                    "phase": phase,
                    "train_episodes": train_episodes,
                    "test_episodes": test_episodes,
                    **evaluation,
                },
            )

        mark_done(
            phase_name,
            {
                "train_episodes": train_episodes,
                "test_episodes": test_episodes,
            },
        )
        print("  Evaluation:", evaluation)


# ============================================================
# FIGURES AND FINAL SUMMARY
# ============================================================

def generate_figures() -> None:
    summary_path = SUMMARY_DIR / "drl_phase_summary.csv"
    if not summary_path.exists():
        print("[Figures] No DRL summary exists yet.")
        return

    frame = pd.read_csv(summary_path)

    final_frame = frame[frame["phase"] == "final"].copy()
    if final_frame.empty:
        return

    grouped = (
        final_frame.groupby("agent", as_index=False)
        .agg(
            mean_reward=("mean_reward", "mean"),
            reward_std=("mean_reward", "std"),
            mean_selected_regions=("mean_selected_regions", "mean"),
            mean_steps=("mean_steps", "mean"),
            mean_classification_loss=(
                "mean_classification_loss", "mean"
            ),
        )
    )

    grouped.to_csv(
        SUMMARY_DIR / "final_agent_summary.csv", index=False
    )

    plt.figure(figsize=(7, 4.5))
    plt.bar(
        grouped["agent"],
        grouped["mean_reward"],
        yerr=grouped["reward_std"].fillna(0.0),
        capsize=4,
    )
    plt.ylabel("Mean evaluation reward")
    plt.xlabel("Agent")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "final_mean_reward.pdf")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.bar(
        grouped["agent"],
        grouped["mean_selected_regions"],
    )
    plt.ylabel("Mean selected regions")
    plt.xlabel("Agent")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "final_selected_regions.pdf")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.bar(
        grouped["agent"],
        grouped["mean_classification_loss"],
    )
    plt.ylabel("Mean classification loss")
    plt.xlabel("Agent")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "final_classification_loss.pdf")
    plt.close()

    print(f"[Figures] Saved in {FIGURE_DIR}")


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark(seed: int = 5) -> None:
    set_seed(seed)
    dataset = ChestXrayDataset(TRAIN_CSV, training=True)
    dataset = Subset(dataset, range(min(128, len(dataset))))
    loader = make_loader(dataset, BASELINE_BATCH_SIZE, True, seed)

    model = DenseNetMultiLabel(pretrained=True).to(device())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_class_weights(TRAIN_CSV).to(device())
    )
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    times: List[float] = []

    print("\n[Benchmark] 128 images, forward + backward")
    for batch_index, batch in enumerate(loader, start=1):
        start = time.perf_counter()
        images = batch["image"].to(device())
        labels = batch["labels"].to(device())

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(
            f"  batch={batch_index}/{len(loader)} "
            f"seconds={elapsed:.2f} loss={loss.item():.4f}"
        )

    seconds_per_batch = float(np.mean(times[2:] or times))
    full_batches = math.ceil(78220 / BASELINE_BATCH_SIZE)
    estimated_epoch_hours = (
        seconds_per_batch * full_batches / 3600.0
    )
    estimated_baseline_hours = estimated_epoch_hours * BASELINE_EPOCHS

    payload = {
        "device": str(device()),
        "cpu_threads": torch.get_num_threads(),
        "batch_size": BASELINE_BATCH_SIZE,
        "seconds_per_batch": seconds_per_batch,
        "estimated_hours_per_full_epoch": estimated_epoch_hours,
        "estimated_hours_for_configured_baseline_epochs": (
            estimated_baseline_hours
        ),
        "note": (
            "Feature caching and DRL phases require additional time. "
            "The estimate is based on the measured baseline batches."
        ),
    }
    save_json(SUMMARY_DIR / "cpu_benchmark.json", payload)
    print("\nBenchmark summary")
    print(json.dumps(payload, indent=4))


# ============================================================
# ORCHESTRATION
# ============================================================

def run_all(
    phase_scale: float,
    max_images: Optional[int],
) -> None:
    ensure_directories()
    verify_required_files()

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device()),
        "torch_version": torch.__version__,
        "configuration": asdict(Config()),
        "phase_scale": phase_scale,
        "max_images": max_images,
        "checkpoint_interval_episodes": CHECKPOINT_INTERVAL_EPISODES,
        "automatic_resume": True,
        "cpu_fast_protocol": {
            "frozen_densenet121_backbone": True,
            "classification_head_epochs": BASELINE_EPOCHS,
            "shared_feature_cache_across_seeds": True,
            "single_backbone_pass_for_global_and_3x3_regions": True,
        },
        "warning": (
            "A phase_scale below 1.0 or max_images value creates a debugging "
            "run and must not be reported as the final paper experiment."
        ),
    }
    save_json(OUTPUT_DIR / "full_experiment_manifest.json", manifest)

    for seed in SEEDS:
        baseline_checkpoint = train_baseline(
            seed,
            max_train_images=max_images,
            max_validation_images=max_images,
        )

        train_cache = cache_features_for_split(
            "train",
            TRAIN_CSV,
            baseline_checkpoint,
            seed,
            max_images=max_images,
        )
        validation_cache = cache_features_for_split(
            "validation",
            VALIDATION_CSV,
            baseline_checkpoint,
            seed,
            max_images=max_images,
        )
        test_cache = cache_features_for_split(
            "test",
            TEST_CSV,
            baseline_checkpoint,
            seed,
            max_images=max_images,
        )

        classifier_checkpoint = train_region_classifier(
            train_cache,
            validation_cache,
            seed,
        )

        for agent_name in ["PPO", "DQN", "DiscreteSAC"]:
            train_agent_phases(
                agent_name,
                seed,
                train_cache,
                test_cache,
                classifier_checkpoint,
                phase_scale=phase_scale,
            )

    generate_figures()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-fast resumable ChestX-ray14 DRL runner (frozen backbone)."
    )
    parser.add_argument(
        "--stage",
        choices=["benchmark", "full", "debug", "figures"],
        default="benchmark",
    )
    parser.add_argument(
        "--phase-scale",
        type=float,
        default=1.0,
        help=(
            "Use 1.0 for the final methodology. Values below 1.0 are "
            "debugging only."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help=(
            "Limit each data split for debugging. Omit for the full dataset."
        ),
    )
    args = parser.parse_args()

    ensure_directories()
    verify_required_files()

    print("=" * 72)
    print("Fast 48h Resumable ChestX-ray14 Region Selection")
    print("=" * 72)
    print(f"Project: {PROJECT_DIR}")
    print(f"Device: {device()}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Stage: {args.stage}")

    if args.stage == "benchmark":
        run_benchmark()
    elif args.stage == "debug":
        run_all(
            phase_scale=0.01,
            max_images=args.max_images or 500,
        )
    elif args.stage == "full":
        if args.phase_scale != 1.0:
            raise ValueError(
                "The full methodology must use --phase-scale 1.0."
            )
        if args.max_images is not None:
            raise ValueError(
                "The full methodology cannot use --max-images."
            )
        run_all(phase_scale=1.0, max_images=None)
    elif args.stage == "figures":
        generate_figures()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Existing CSV logs and checkpoints remain.")
        raise SystemExit(130)
    except Exception as error:
        print(f"\nFAILED: {error}", file=sys.stderr)
        raise
