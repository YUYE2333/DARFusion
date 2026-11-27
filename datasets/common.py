from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_cdt
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from random_control import derived_seed
from .spec import DatasetSpec


@dataclass
class DatasetBundle:
    """Normalized co-registered scene and its configured label split maps."""

    hsi: np.ndarray
    lidar: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    gt: Optional[np.ndarray]
    num_bands: int
    lidar_channels: int
    num_classes: int
    norm_stats: Dict[str, np.ndarray]
    dataset_name: str = ""
    class_names: Tuple[str, ...] = ()
    metadata: Dict[str, object] = field(default_factory=dict)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int, *, seed: int) -> None:
    worker_seed = derived_seed(seed, "worker", worker_id)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reset_loader_random_streams(
    loaders: Dict[str, DataLoader], seed: int
) -> None:
    generator = getattr(loaders.get("train"), "generator", None)
    if generator is not None:
        generator.manual_seed(derived_seed(seed, "dataloader"))


def split_protocol_metadata(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    patch_size: int,
) -> Dict[str, object]:
    train_mask = train_labels > 0
    test_mask = test_labels > 0
    train_count = int(train_mask.sum())
    test_count = int(test_mask.sum())
    if train_count <= 0 or test_count <= 0:
        raise ValueError("Train/test maps must both contain labeled pixels.")

    center_overlap = int(np.sum(train_mask & test_mask))
    if center_overlap:
        raise ValueError(
            "Train/test label maps overlap at "
            f"{center_overlap} center pixel(s)."
        )

    # This is an audit only. Standard benchmark samples are never filtered.
    center_distances = distance_transform_cdt(~train_mask, metric="chessboard")
    test_distances = center_distances[test_mask]
    overlap_radius = patch_size - 1
    patch_overlap_count = int(np.sum(test_distances <= overlap_radius))
    return {
        "protocol": "single_mat_label_split",
        "train_source": "configured split MAT train key",
        "test_source": "configured split MAT evaluation key",
        "train_samples": train_count,
        "test_samples": test_count,
        "train_test_center_disjoint": True,
        "patch_overlap_policy": "audit_only_no_filtering",
        "patch_size": int(patch_size),
        "overlap_center_radius": int(overlap_radius),
        "test_patches_overlapping_any_train_patch": patch_overlap_count,
        "test_patch_overlap_fraction": float(patch_overlap_count / test_count),
    }


class MultimodalPatchDataset(Dataset):
    def __init__(
        self,
        hsi: np.ndarray,
        lidar: np.ndarray,
        label_map: np.ndarray,
        coordinates: np.ndarray,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.label_map = label_map
        self.coordinates = coordinates.astype(np.int64, copy=False)
        # The loader shares the original scene arrays and pads only boundary patches
        # on demand, avoiding unnecessary duplication of the complete scene.
        self.hsi = hsi
        self.lidar = lidar

    def _extract_patch(self, array: np.ndarray, row: int, col: int) -> np.ndarray:
        height, width = array.shape[:2]
        row0, row1 = row - self.pad, row + self.pad + 1
        col0, col1 = col - self.pad, col + self.pad + 1
        src = array[
            max(0, row0):min(height, row1),
            max(0, col0):min(width, col1),
        ]
        padding = (
            (max(0, -row0), max(0, row1 - height)),
            (max(0, -col0), max(0, col1 - width)),
            (0, 0),
        )
        if any(before or after for before, after in padding[:2]):
            src = np.pad(src, padding, mode="reflect")
        return src

    def __len__(self) -> int:
        return self.coordinates.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        row, col = self.coordinates[index]
        hsi_patch = self._extract_patch(self.hsi, int(row), int(col))
        lidar_patch = self._extract_patch(self.lidar, int(row), int(col))
        label = int(self.label_map[row, col]) - 1
        return {
            "hsi": torch.from_numpy(
                np.ascontiguousarray(hsi_patch.transpose(2, 0, 1))
            ).float(),
            "lidar": torch.from_numpy(
                np.ascontiguousarray(lidar_patch.transpose(2, 0, 1))
            ).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "row": torch.tensor(row, dtype=torch.long),
            "col": torch.tensor(col, dtype=torch.long),
        }


def make_loaders(
    bundle: DatasetBundle,
    spec: DatasetSpec,
) -> Dict[str, DataLoader]:
    train_coordinates = np.argwhere(bundle.train_labels > 0)
    test_coordinates = np.argwhere(bundle.test_labels > 0)
    protocol = split_protocol_metadata(
        bundle.train_labels, bundle.test_labels, spec.patch_size
    )
    split_meta = dict(bundle.metadata.get("split", {}))
    evaluation_split = str(split_meta.get("evaluation_split", "configured_test"))
    protocol["evaluation_split"] = evaluation_split
    bundle.metadata["sampling_protocol"] = protocol
    print(
        f"[data-protocol] train={len(train_coordinates)}, {evaluation_split}={len(test_coordinates)}, "
        "center_disjoint=true, patch_overlap=audit_only"
    )

    datasets = {
        "train": MultimodalPatchDataset(
            bundle.hsi,
            bundle.lidar,
            bundle.train_labels,
            train_coordinates,
            spec.patch_size,
        ),
        "test": MultimodalPatchDataset(
            bundle.hsi,
            bundle.lidar,
            bundle.test_labels,
            test_coordinates,
            spec.patch_size,
        ),
    }
    common = {
        "batch_size": spec.batch_size,
        "num_workers": spec.num_workers,
        "pin_memory": spec.pin_memory and torch.cuda.is_available(),
        "worker_init_fn": partial(_seed_worker, seed=spec.seed),
    }
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(derived_seed(spec.seed, "dataloader"))
    return {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            drop_last=False,
            generator=train_generator,
            **common,
        ),
        "test": DataLoader(
            datasets["test"], shuffle=False, drop_last=False, **common
        ),
    }


__all__ = [
    "DatasetBundle",
    "MultimodalPatchDataset",
    "make_loaders",
    "split_protocol_metadata",
    "reset_loader_random_streams",
    "seed_everything",
]

