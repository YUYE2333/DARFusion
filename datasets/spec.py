from __future__ import annotations

"""Typed runtime settings for the configured single-MAT split pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from random_control import validate_seed


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    loader_name: str
    data_root: Path
    patch_size: int
    seed: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    config_path: Path
    config_sha1: str

    @classmethod
    def from_config_bundle(
        cls,
        bundle: Mapping[str, Any],
        data_root_override: Optional[Path] = None,
    ) -> "DatasetSpec":
        dataset = dict(bundle["dataset"])
        main = dict(bundle["main"])
        runtime = dict(main["runtime"])
        experiment = dict(main["experiment"])
        seeds = dict(main.get("seeds", {}))
        paths = dict(bundle["paths"])
        hashes = dict(bundle["hashes"])
        root = (
            Path(data_root_override).expanduser().resolve()
            if data_root_override is not None
            else Path(str(paths["dataset_root"])).expanduser().resolve()
        )
        spec = cls(
            dataset_name=str(dataset["dataset"]),
            loader_name=str(dataset["loader"]),
            data_root=root,
            patch_size=int(dataset["patch_size"]),
            seed=validate_seed(seeds.get("model", experiment["seed"])),
            batch_size=int(runtime["batch_size"]),
            num_workers=int(runtime["num_workers"]),
            pin_memory=bool(runtime["pin_memory"]),
            config_path=Path(str(paths["dataset_config"])),
            config_sha1=str(hashes["dataset_config_sha1"]),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.dataset_name or not self.loader_name:
            raise ValueError("Dataset name and loader name must not be empty.")
        if self.patch_size <= 0 or self.patch_size % 2 != 1:
            raise ValueError("Dataset patch_size must be a positive odd integer.")
        if self.batch_size <= 0:
            raise ValueError("runtime.batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("runtime.num_workers must be non-negative.")


__all__ = ["DatasetSpec"]
