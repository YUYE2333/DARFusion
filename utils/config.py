from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha1(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _resolve_relative(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def _project_root(config_path: Path) -> Path:
    """Locate the portable PCRFusion root from any config below ``configs/``."""
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "configs").is_dir() and (parent / "train.py").is_file():
            return parent.resolve()
    raise RuntimeError(f"Cannot locate PCRFusion project root from config: {config_path}")


def load_experiment(
    config_path: Path,
    *,
    data_root_override: Optional[Path] = None,
    split_file_override: Optional[Path] = None,
    output_root_override: Optional[Path] = None,
    model_seed_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve one PCRFusion experiment JSON and its referenced dataset/training/degradation JSONs."""
    config_path = Path(config_path).expanduser().resolve()
    main = copy.deepcopy(_read_json(config_path))
    base = config_path.parent
    project_root = _project_root(config_path)

    dataset_path = _resolve_relative(base, str(main["dataset_config"]))
    degradation_path = _resolve_relative(base, str(main["degradation_config"]))
    training_path = _resolve_relative(base, str(main["training_config"]))
    dataset = _read_json(dataset_path)
    degradation = _read_json(degradation_path)
    training = _read_json(training_path)

    if "label_split" not in dataset:
        raise ValueError(
            f"dataset JSON must define one label_split MAT file: {dataset_path}"
        )
    label_split = copy.deepcopy(dict(dataset["label_split"]))
    if split_file_override is not None:
        split_file = Path(split_file_override).expanduser().resolve()
    else:
        split_file = _resolve_relative(project_root, str(label_split["file"]))
    label_split["file"] = str(split_file)
    dataset["label_split"] = label_split

    # Shared, dataset-agnostic training parameters are loaded from configs/training/*.json.
    # Experiments remain responsible only for dataset/protocol/fold/validation/evaluation choices.
    for key in ("runtime", "clean_fusion", "robust_fusion"):
        if key not in training:
            raise ValueError(f"training JSON must define '{key}'")
        if key in main:
            raise ValueError(
                f"'{key}' must not be duplicated in experiment JSON; put it in {training_path}"
            )
        main[key] = copy.deepcopy(training[key])

    # The benchmark keeps the configured train/test maps intact. Unknown
    # Gaussian/Holes are never used by a training stage.  Checkpoint selection
    # remains explicit in the resolved metadata so a paper table cannot silently
    # mix protocols.
    main.pop("mode", None)
    main.pop("train_target", None)
    main.setdefault("validation", {"enabled": False})
    main.setdefault(
        "selection_protocol",
        "development_validation" if bool(main["validation"].get("enabled", False)) else "official_test_guided",
    )
    main.setdefault("seeds", {})
    main["seeds"].setdefault("split", 2026)
    main["seeds"].setdefault("model", 1234)
    main["seeds"].setdefault("corruption", 1234)
    main.setdefault("runtime", {})

    if model_seed_override is not None:
        main["seeds"]["model"] = int(model_seed_override)
    if batch_size_override is not None:
        main["runtime"]["batch_size"] = int(batch_size_override)
    if device_override is not None:
        main["runtime"]["device"] = str(device_override)

    folds = list(main.get("held_out_folds", []))
    if not folds:
        raise ValueError("experiment JSON must define at least one held_out_folds entry")
    active = int(main.get("active_fold", 0))
    if not 0 <= active < len(folds):
        raise ValueError(f"active_fold={active} is outside held_out_folds[0:{len(folds)}]")
    fold = dict(folds[active])
    if "hsi" not in fold or "lidar" not in fold:
        raise ValueError("Each held_out_folds entry must define hsi and lidar")

    if data_root_override is not None:
        data_root = Path(data_root_override).expanduser().resolve()
    else:
        default_root = Path(str(dataset["default_root"]))
        data_root = default_root.resolve() if default_root.is_absolute() else (project_root / default_root).resolve()

    if output_root_override is not None:
        output_root = Path(output_root_override).expanduser().resolve()
    else:
        raw = Path(str(main.get("output_root", "outputs"))).expanduser()
        output_root = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()

    experiment = {
        "name": str(main.get("name", dataset["dataset"])),
        "active_fold": active,
        "fold_name": str(fold.get("name", f"fold{active + 1}")),
        "hsi_outer_unknown": str(fold["hsi"]),
        "lidar_outer_unknown": str(fold["lidar"]),
        "seed": int(main["seeds"]["model"]),  # shared seed used by dataset and training engines
    }

    runtime_view = {
        "runtime": main["runtime"],
        "experiment": experiment,
        "validation": main["validation"],
        "seeds": main["seeds"],
    }

    return {
        "config_path": str(config_path),
        "config": main,
        "main": runtime_view,
        "dataset": dataset,
        "training": training,
        "degradation": degradation,
        "experiment": experiment,
        "paths": {
            "project_root": str(project_root),
            "dataset_root": str(data_root),
            "label_split_file": str(split_file),
            "output_root": str(output_root),
            "dataset_config": str(dataset_path),
            "degradation_config": str(degradation_path),
            "training_config": str(training_path),
        },
        "hashes": {
            "experiment_config_sha1": _sha1(main),
            "dataset_config_sha1": _sha1(dataset),
            "degradation_config_sha1": _sha1(degradation),
            "training_config_sha1": _sha1(training),
        },
    }
