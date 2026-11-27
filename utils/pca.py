from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
from torch import Tensor, nn


def fit_training_pca(hsi: np.ndarray, train_labels: np.ndarray, n_components: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA from configured training-label pixels only.

    The HSI passed here is the normalized *pre-PCA* scene.  Corruptions remain
    defined in this pre-PCA space; the learned transform is then reused for all
    clean/degraded inputs.
    """
    mask = np.asarray(train_labels) > 0
    x = np.asarray(hsi, dtype=np.float64)[mask]
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("PCA requires at least two labeled training pixels")
    k = min(int(n_components), int(x.shape[1]), int(x.shape[0]))
    if k <= 0:
        raise ValueError("PCA n_components must be positive")
    mean = x.mean(axis=0)
    xc = x - mean
    # SVD on training pixels only.  Vt rows are orthonormal principal axes.
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:k]
    denom = max(x.shape[0] - 1, 1)
    explained = (s[:k] ** 2) / denom
    total = float((s ** 2).sum() / denom)
    ratio = explained / max(total, 1e-12)
    return mean.astype(np.float32), components.astype(np.float32), ratio.astype(np.float32)


class FixedPCATransform(nn.Module):
    """Frozen linear PCA transform for BxCxHxW HSI patches."""

    def __init__(self, in_bands: int, out_bands: int) -> None:
        super().__init__()
        self.in_bands = int(in_bands)
        self.out_bands = int(out_bands)
        self.register_buffer("mean", torch.zeros(self.in_bands))
        self.register_buffer("components", torch.zeros(self.out_bands, self.in_bands))
        self.register_buffer("configured", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def set_parameters(self, mean: np.ndarray | Tensor, components: np.ndarray | Tensor) -> None:
        mean_t = torch.as_tensor(mean, dtype=self.mean.dtype, device=self.mean.device)
        comp_t = torch.as_tensor(components, dtype=self.components.dtype, device=self.components.device)
        if tuple(mean_t.shape) != tuple(self.mean.shape):
            raise ValueError(f"PCA mean shape {tuple(mean_t.shape)} != {tuple(self.mean.shape)}")
        if tuple(comp_t.shape) != tuple(self.components.shape):
            raise ValueError(f"PCA components shape {tuple(comp_t.shape)} != {tuple(self.components.shape)}")
        self.mean.copy_(mean_t)
        self.components.copy_(comp_t)
        self.configured.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_bands:
            raise ValueError(f"Expected HSI Bx{self.in_bands}xHxW, got {tuple(x.shape)}")
        if not bool(self.configured.item()):
            raise RuntimeError("Fixed PCA transform has not been configured")
        centered = x - self.mean[None, :, None, None]
        return torch.einsum("bchw,kc->bkhw", centered, self.components)
