from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from random_control import derived_seed
from utils.resume import atomic_numpy
from .operators import apply_cross_modal_corruption, apply_hsi_corruption, apply_lidar_corruption
from .protocol import DualModalDegradationProtocol


def _reflect_indices(indices: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros_like(indices)
    period = 2 * size - 2
    folded = np.mod(indices, period)
    return np.where(folded < size, folded, period - folded)


class DualModalWholeSceneProvider:
    """Deterministic whole-scene dual-modal corruption provider.

    HSI, LiDAR and relation-level corruptions use independent derived RNG
    streams from one master seed.  Cross-modal misregistration is applied to
    LiDAR *after* any LiDAR-local corruption so the condition is reproducible.
    """

    def __init__(
        self,
        clean_hsi: np.ndarray,
        clean_lidar: np.ndarray,
        cache_root: Path,
        patch_size: int,
        protocol: DualModalDegradationProtocol,
        seed: int,
    ) -> None:
        if clean_hsi.ndim != 3 or clean_lidar.ndim != 3:
            raise ValueError("clean_hsi/clean_lidar must be [H,W,C].")
        if clean_hsi.shape[:2] != clean_lidar.shape[:2]:
            raise ValueError("HSI/LiDAR spatial shapes must match.")
        self.clean_hsi = clean_hsi.astype(np.float32, copy=False)
        self.clean_lidar = clean_lidar.astype(np.float32, copy=False)
        self.height, self.width = clean_hsi.shape[:2]
        self.patch_size = int(patch_size)
        self.pad = self.patch_size // 2
        self.protocol = protocol
        self.seed = int(seed)
        self.cache_dtype = np.dtype(protocol.cache_dtype)
        payload = {
            "seed": self.seed,
            "hsi_shape": list(clean_hsi.shape),
            "lidar_shape": list(clean_lidar.shape),
            "protocol_sha1": protocol.config_sha1,
        }
        fingerprint = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        self.cache_dir = Path(cache_root) / fingerprint
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[dual-degradation-cache] {self.cache_dir}")

    def _key(self, modality: str, name: str, severity: int) -> str:
        return f"{modality}__{name}__s{int(severity)}"

    def _rng(self, key: str) -> np.random.Generator:
        return np.random.default_rng(derived_seed(self.seed, f"corruption:{key}"))

    def get_hsi_scene(self, name: Optional[str], severity: int = 0) -> np.ndarray:
        if name is None:
            return self.clean_hsi
        key = self._key("hsi", name, severity)
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path, mmap_mode="r")
        print(f"[dual-degradation-cache] generating {key}")
        out = apply_hsi_corruption(
            self.clean_hsi, name, self.protocol.hsi_params(name, severity), self._rng(key)
        )
        atomic_numpy(path, np.ascontiguousarray(out.astype(self.cache_dtype, copy=False)))
        return np.load(path, mmap_mode="r")

    def get_lidar_scene(self, name: Optional[str], severity: int = 0) -> np.ndarray:
        if name is None:
            return self.clean_lidar
        key = self._key("lidar", name, severity)
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path, mmap_mode="r")
        print(f"[dual-degradation-cache] generating {key}")
        out = apply_lidar_corruption(
            self.clean_lidar, name, self.protocol.lidar_params(name, severity), self._rng(key)
        )
        atomic_numpy(path, np.ascontiguousarray(out.astype(self.cache_dtype, copy=False)))
        return np.load(path, mmap_mode="r")

    def get_cross_modal_lidar_scene(
        self,
        lidar_name: Optional[str],
        lidar_severity: int,
        cross_name: Optional[str],
        cross_severity: int,
    ) -> np.ndarray:
        base = self.get_lidar_scene(lidar_name, lidar_severity)
        if cross_name is None:
            return base
        local_key = "clean" if lidar_name is None else f"{lidar_name}-s{int(lidar_severity)}"
        key = f"cross__{cross_name}__s{int(cross_severity)}__on__{local_key}"
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path, mmap_mode="r")
        print(f"[dual-degradation-cache] generating {key}")
        out = apply_cross_modal_corruption(
            np.asarray(base, dtype=np.float32),
            cross_name,
            self.protocol.cross_modal_params(cross_name, cross_severity),
            self._rng(key),
        )
        atomic_numpy(path, np.ascontiguousarray(out.astype(self.cache_dtype, copy=False)))
        return np.load(path, mmap_mode="r")

    def _extract(self, scene: np.ndarray, row: int, col: int) -> np.ndarray:
        rows = _reflect_indices(np.arange(row - self.pad, row + self.pad + 1), self.height)
        cols = _reflect_indices(np.arange(col - self.pad, col + self.pad + 1), self.width)
        patch = scene[rows[:, None], cols[None, :], :]
        return np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32)

    def extract_condition(
        self,
        rows: Tensor,
        cols: Tensor,
        device: torch.device,
        hsi_name: Optional[str] = None,
        hsi_severity: int = 0,
        lidar_name: Optional[str] = None,
        lidar_severity: int = 0,
        cross_name: Optional[str] = None,
        cross_severity: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        hs = self.get_hsi_scene(hsi_name, hsi_severity)
        ls = self.get_cross_modal_lidar_scene(
            lidar_name, lidar_severity, cross_name, cross_severity
        )
        rr = rows.detach().cpu().numpy().astype(np.int64)
        cc = cols.detach().cpu().numpy().astype(np.int64)
        h = np.stack([self._extract(hs, int(r), int(c)) for r, c in zip(rr, cc)])
        l = np.stack([self._extract(ls, int(r), int(c)) for r, c in zip(rr, cc)])
        return (
            torch.from_numpy(h).to(device, non_blocking=True),
            torch.from_numpy(l).to(device, non_blocking=True),
        )

    def sample_training_batch(
        self,
        rows: Tensor,
        cols: Tensor,
        device: torch.device,
        rng: np.random.Generator,
        states: Sequence[str],
        probabilities: Sequence[float],
        hsi_known: Sequence[str],
        lidar_known: Sequence[str],
    ):
        """Sample modality-local corruptions only.

        The default protocol leaves misregistration out of training so it can serve as
        a relation-level stress test for direct vs UOT shared alignment.
        """
        batch = int(rows.numel())
        state_ids = rng.choice(len(states), size=batch, p=np.asarray(probabilities, dtype=np.float64))
        h_names: List[Optional[str]] = [None] * batch
        l_names: List[Optional[str]] = [None] * batch
        h_sev = np.zeros(batch, dtype=np.int64)
        l_sev = np.zeros(batch, dtype=np.int64)
        for i, sid in enumerate(state_ids):
            state = str(states[int(sid)])
            if state in {"hsi_degraded", "dual_degraded"}:
                h_names[i] = str(hsi_known[int(rng.integers(0, len(hsi_known)))])
                h_sev[i] = int(rng.choice(self.protocol.severity_levels))
            if state in {"lidar_degraded", "dual_degraded"}:
                l_names[i] = str(lidar_known[int(rng.integers(0, len(lidar_known)))])
                l_sev[i] = int(rng.choice(self.protocol.severity_levels))
        rr = rows.detach().cpu().numpy().astype(np.int64)
        cc = cols.detach().cpu().numpy().astype(np.int64)
        groups: Dict[Tuple[Optional[str], int, Optional[str], int], List[int]] = {}
        for i in range(batch):
            groups.setdefault((h_names[i], int(h_sev[i]), l_names[i], int(l_sev[i])), []).append(i)
        h_out = None
        l_out = None
        for (hn, hs, ln, ls), idxs in groups.items():
            hscene = self.get_hsi_scene(hn, hs)
            lscene = self.get_lidar_scene(ln, ls)
            ha = np.stack([self._extract(hscene, int(rr[i]), int(cc[i])) for i in idxs])
            la = np.stack([self._extract(lscene, int(rr[i]), int(cc[i])) for i in idxs])
            if h_out is None:
                h_out = torch.empty((batch, *ha.shape[1:]), device=device, dtype=torch.float32)
                l_out = torch.empty((batch, *la.shape[1:]), device=device, dtype=torch.float32)
            tidx = torch.tensor(idxs, dtype=torch.long, device=device)
            h_out.index_copy_(0, tidx, torch.from_numpy(ha).to(device))
            l_out.index_copy_(0, tidx, torch.from_numpy(la).to(device))
        return h_out, l_out
