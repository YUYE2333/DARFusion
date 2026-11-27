from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

from .common import DatasetBundle
from .mat_hsi_lidar import load_mat_hsi_lidar


Loader = Callable[..., DatasetBundle]
LOADERS: Dict[str, Loader] = {
    "mat_hsi_lidar": load_mat_hsi_lidar,
}


def register_dataset_loader(name: str, loader: Loader) -> None:
    if not name or name in LOADERS:
        raise ValueError(f"Dataset loader {name!r} is empty or already registered.")
    LOADERS[name] = loader


def load_dataset(
    config_bundle: Mapping[str, object],
    data_root: Optional[Path] = None,
) -> DatasetBundle:
    dataset_config = dict(config_bundle["dataset"])
    loader_name = str(dataset_config["loader"])
    if loader_name not in LOADERS:
        raise KeyError(
            f"Unknown dataset loader {loader_name!r}; available={sorted(LOADERS)}"
        )
    root = Path(data_root) if data_root is not None else Path(str(config_bundle["paths"]["dataset_root"]))
    main = dict(config_bundle.get("main", {}))
    validation = dict(main.get("validation", {}))
    seeds = dict(main.get("seeds", {}))
    split_seed = seeds.get("split")
    return LOADERS[loader_name](
        dataset_config,
        root.expanduser().resolve(),
        validation=validation,
        split_seed=None if split_seed is None else int(split_seed),
    )


__all__ = ["LOADERS", "load_dataset", "register_dataset_loader"]
