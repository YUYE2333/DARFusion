from __future__ import annotations

"""One config-driven MAT loader for co-registered HSI and auxiliary rasters."""

from itertools import permutations
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.io import loadmat

from .common import DatasetBundle


def _numeric_arrays_from_scipy(data: Dict[str, object]) -> Dict[str, np.ndarray]:
    return {
        key: value
        for key, value in data.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
    }


def _load_hdf5_arrays(
    path: Path,
    requested_keys: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    requested = None if requested_keys is None else {str(key) for key in requested_keys}
    with h5py.File(path, "r") as handle:

        def visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                leaf_name = name.split("/")[-1]
                if requested is not None and leaf_name not in requested:
                    return
                value = np.asarray(obj)
                if np.issubdtype(value.dtype, np.number):
                    if value.ndim > 1:
                        value = value.transpose(tuple(reversed(range(value.ndim))))
                    result[leaf_name] = value

        handle.visititems(visit)
    return result


def _as_candidates(value: object, field: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        candidates = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        candidates = tuple(str(item) for item in value)
    else:
        raise ValueError(f"{field} must be a string or a list of strings.")
    candidates = tuple(item for item in candidates if item)
    if not candidates:
        raise ValueError(f"{field} contains no candidates.")
    return candidates


class MatArrayStore:
    """Resolve file/key candidates and avoid reloading shared MAT containers."""

    def __init__(
        self,
        data_root: Path,
        *,
        allow_implicit_key_fallback: bool = False,
    ) -> None:
        self.data_root = data_root
        self.allow_implicit_key_fallback = bool(allow_implicit_key_fallback)
        self._arrays: Dict[Path, Dict[str, np.ndarray]] = {}
        self._fully_loaded: set[Path] = set()
        self.used_files: Dict[str, str] = {}
        self.used_keys: Dict[str, str] = {}

    def _resolve_file(self, spec: object, role: str, required: bool) -> Optional[Path]:
        candidates = _as_candidates(spec, f"files.{role}")
        for candidate in candidates:
            path = (self.data_root / candidate).resolve()
            if path.exists():
                return path
        if required:
            listed = ", ".join(str(self.data_root / value) for value in candidates)
            raise FileNotFoundError(f"Missing {role} MAT file; tried: {listed}")
        return None

    def _read_file(
        self,
        path: Path,
        requested_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        cached = self._arrays.setdefault(path, {})
        requested = None if requested_keys is None else tuple(str(x) for x in requested_keys)
        if requested is None and path in self._fully_loaded:
            return cached
        if requested is not None and any(key in cached for key in requested):
            return cached
        try:
            arrays = _numeric_arrays_from_scipy(
                loadmat(path, variable_names=None if requested is None else list(requested))
            )
        except (NotImplementedError, ValueError):
            arrays = _load_hdf5_arrays(path, requested)
        cached.update(arrays)
        if requested is None:
            self._fully_loaded.add(path)
        if requested is None and not cached:
            raise ValueError(f"No numeric MATLAB arrays found in {path}")
        return cached

    def load(
        self,
        file_spec: object,
        key_spec: object,
        role: str,
        expected_ndim: Optional[int],
        *,
        required: bool = True,
    ) -> Optional[np.ndarray]:
        path = self._resolve_file(file_spec, role, required)
        if path is None:
            return None
        key_candidates = _as_candidates(key_spec, f"keys.{role}")
        arrays = self._read_file(path, key_candidates)
        selected_key = next((key for key in key_candidates if key in arrays), None)
        if selected_key is None:
            if not self.allow_implicit_key_fallback:
                if not required:
                    return None
                raise KeyError(
                    f"{path.name}: none of the configured keys {key_candidates!r} "
                    f"exists for role {role!r}. Available numeric keys are "
                    f"{sorted(arrays)}. Correct keys.{role} in the dataset JSON; "
                    "implicit largest-array guessing is disabled."
                )
            arrays = self._read_file(path, None)
            candidates = list(arrays.items())
            if expected_ndim is not None:
                matching = [
                    item
                    for item in candidates
                    if np.squeeze(item[1]).ndim == expected_ndim
                ]
                if matching:
                    candidates = matching
            selected_key, array = max(candidates, key=lambda item: item[1].size)
            print(
                f"[data] {path.name}: none of keys {key_candidates!r} found; "
                f"using {selected_key!r} with shape {array.shape}."
            )
        else:
            array = arrays[selected_key]
        self.used_files[role] = str(path)
        self.used_keys[role] = str(selected_key)
        return np.asarray(array)


def _expected_spatial_shape(expected: Mapping[str, object]) -> Optional[Tuple[int, int]]:
    value = expected.get("spatial_shape")
    if value is None:
        return None
    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("expected.spatial_shape must contain [height, width].")
    return int(value[0]), int(value[1])


def align_label(
    array: np.ndarray,
    name: str,
    layout: str = "auto",
    expected_spatial: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D after squeeze, got {array.shape}.")
    layout = layout.upper()
    if layout == "WH":
        array = array.T
    elif layout not in {"AUTO", "HW"}:
        raise ValueError(f"Unsupported label layout {layout!r}; use auto, HW or WH.")
    elif layout == "AUTO" and expected_spatial is not None:
        if array.shape == expected_spatial:
            pass
        elif array.T.shape == expected_spatial:
            array = array.T
        else:
            raise ValueError(
                f"{name} shape {array.shape} cannot match declared spatial shape "
                f"{expected_spatial}."
            )
    return array.astype(np.int64, copy=False)


def _layout_cube(array: np.ndarray, layout: str, role: str) -> np.ndarray:
    layout = layout.upper()
    aliases = {
        "HWC": (0, 1, 2),
        "HWB": (0, 1, 2),
        "CHW": (1, 2, 0),
        "BHW": (1, 2, 0),
        "WHC": (1, 0, 2),
        "WHB": (1, 0, 2),
        "CWH": (2, 1, 0),
        "BWH": (2, 1, 0),
    }
    if layout not in aliases:
        raise ValueError(
            f"Unsupported {role} layout {layout!r}; use auto, HWC/HWB, "
            "CHW/BHW, WHC/WHB or CWH/BWH."
        )
    return array.transpose(aliases[layout])


def align_cube(
    array: np.ndarray,
    height: int,
    width: int,
    role: str,
    layout: str = "auto",
) -> np.ndarray:
    # Preserve an explicit singleton channel axis (e.g. MUUFL HWC LiDAR).
    if not (array.ndim == 3 and layout.upper() not in {"HW", "WH"}):
        array = np.squeeze(array)
    if array.ndim == 2:
        if role == "hsi":
            raise ValueError(f"HSI must be 3-D, got shape {array.shape}.")
        if layout.upper() == "WH":
            array = array.T
        elif layout.upper() not in {"AUTO", "HW"}:
            raise ValueError(f"Unsupported 2-D {role} layout {layout!r}.")
        if array.shape == (width, height) and array.shape != (height, width):
            array = array.T
        if array.shape != (height, width):
            raise ValueError(
                f"Cannot align {role} shape {array.shape} to labels {(height, width)}."
            )
        return array[..., None].astype(np.float32, copy=False)
    if array.ndim != 3:
        raise ValueError(f"{role} must be 2-D or 3-D, got shape {array.shape}.")
    if layout.upper() != "AUTO":
        candidate = _layout_cube(array, layout, role)
        if candidate.shape[:2] != (height, width):
            raise ValueError(
                f"Configured {role} layout {layout!r} produces {candidate.shape}, "
                f"not spatial shape {(height, width)}."
            )
        return candidate.astype(np.float32, copy=False)
    matches: List[np.ndarray] = []
    for order in permutations(range(3)):
        candidate = array.transpose(order)
        if candidate.shape[:2] == (height, width):
            matches.append(candidate)
    if not matches:
        raise ValueError(f"Cannot align {role} shape {array.shape} to {(height, width)}.")
    # When more than one ordering matches a square scene, prefer the candidate
    # with the smallest last dimension as the modality-channel axis.
    candidate = min(matches, key=lambda value: value.shape[2])
    return candidate.astype(np.float32, copy=False)


def apply_band_selection(hsi: np.ndarray, spec: Mapping[str, object]) -> np.ndarray:
    mode = str(spec.get("mode", "all")).lower()
    if mode == "all":
        return hsi
    if mode == "slice":
        start = int(spec.get("start", 0))
        stop_value = spec.get("stop")
        stop = None if stop_value is None else int(stop_value)
        step = int(spec.get("step", 1))
        result = hsi[..., slice(start, stop, step)]
    elif mode == "indices":
        indices = np.asarray(spec.get("indices", ()), dtype=np.int64)
        if indices.size == 0:
            raise ValueError("band_selection.indices must not be empty.")
        result = hsi[..., indices]
    elif mode == "drop_indices":
        dropped = {int(value) for value in spec.get("indices", ())}
        indices = [index for index in range(hsi.shape[2]) if index not in dropped]
        result = hsi[..., indices]
    else:
        raise ValueError(
            "band_selection.mode must be all, slice, indices or drop_indices."
        )
    if result.shape[2] == 0:
        raise ValueError("Band selection removed every HSI band.")
    return np.ascontiguousarray(result, dtype=np.float32)


def _replace_label_values(
    array: np.ndarray, replacements: Mapping[str, object]
) -> np.ndarray:
    if not replacements:
        return array
    result = array.copy()
    for source, target in replacements.items():
        result[array == int(source)] = int(target)
    return result



def stratified_split_label_map(
    train_labels: np.ndarray,
    val_ratio: float,
    seed: int,
    *,
    min_train_per_class: int = 1,
    min_val_per_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Split the configured training label map into train-dev/validation.

    Splitting is class-stratified and deterministic for a fixed split seed.
    The fixed test key is not selected and is not returned here.
    """
    ratio = float(val_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("validation.ratio must be between 0 and 1")
    rng = np.random.default_rng(int(seed))
    dev = np.zeros_like(train_labels)
    val = np.zeros_like(train_labels)
    per_class = {}
    classes = [int(x) for x in np.unique(train_labels) if int(x) > 0]
    for cls in classes:
        coords = np.argwhere(train_labels == cls)
        n = len(coords)
        if n < min_train_per_class + min_val_per_class:
            raise ValueError(
                f"Class {cls} has only {n} configured-train samples; cannot reserve "
                f"min_train={min_train_per_class}, min_val={min_val_per_class}."
            )
        order = rng.permutation(n)
        n_val = int(round(n * ratio))
        n_val = max(int(min_val_per_class), n_val)
        n_val = min(n - int(min_train_per_class), n_val)
        val_coords = coords[order[:n_val]]
        dev_coords = coords[order[n_val:]]
        val[val_coords[:, 0], val_coords[:, 1]] = cls
        dev[dev_coords[:, 0], dev_coords[:, 1]] = cls
        per_class[str(cls)] = {"train": int(len(dev_coords)), "validation": int(len(val_coords))}
    meta = {
        "strategy": "stratified_random",
        "ratio": ratio,
        "split_seed": int(seed),
        "train_samples": int((dev > 0).sum()),
        "validation_samples": int((val > 0).sum()),
        "per_class": per_class,
    }
    return dev, val, meta


def normalize_scene(
    hsi: np.ndarray,
    lidar: np.ndarray,
    train_labels: np.ndarray,
    percentile_low: float,
    percentile_high: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Train-label-only normalization shared by every configured dataset."""
    train_mask = train_labels > 0
    if not np.any(train_mask):
        raise ValueError("Training label map contains no positive labels.")
    train_hsi = hsi[train_mask]
    low = np.percentile(train_hsi, percentile_low, axis=0).astype(np.float32)
    high = np.percentile(train_hsi, percentile_high, axis=0).astype(np.float32)
    scale = np.maximum(high - low, 1e-6)
    hsi_norm = np.clip((hsi - low) / scale, 0.0, 1.0).astype(np.float32)
    train_lidar = lidar[train_mask]
    lidar_mean = train_lidar.mean(axis=0, dtype=np.float64).astype(np.float32)
    lidar_std = train_lidar.std(axis=0, dtype=np.float64).astype(np.float32)
    lidar_std = np.maximum(lidar_std, 1e-6)
    lidar_norm = ((lidar - lidar_mean) / lidar_std).astype(np.float32)
    return hsi_norm, lidar_norm, {
        "hsi_low": low,
        "hsi_high": high,
        "lidar_mean": lidar_mean,
        "lidar_std": lidar_std,
    }


def _matches_declared(actual: int, declared: object) -> bool:
    if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
        return actual in {int(value) for value in declared}
    return actual == int(declared)


def _validate_expected(
    bundle: DatasetBundle,
    expected: Mapping[str, object],
) -> None:
    checks = {
        "num_bands": bundle.num_bands,
        "lidar_channels": bundle.lidar_channels,
        "num_classes": bundle.num_classes,
    }
    for key, actual in checks.items():
        declared = expected.get(key)
        if declared is not None and not _matches_declared(int(actual), declared):
            raise ValueError(
                f"Dataset config declares {key}={declared}, but loaded data has {actual}."
            )
    declared_spatial = _expected_spatial_shape(expected)
    if declared_spatial is not None and bundle.hsi.shape[:2] != declared_spatial:
        raise ValueError(
            f"Dataset config declares spatial_shape={declared_spatial}, but loaded "
            f"data has {bundle.hsi.shape[:2]}."
        )


def _validate_label_schema(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    schema: Mapping[str, object],
) -> None:
    train_ids = {int(value) for value in np.unique(train_labels) if int(value) > 0}
    test_ids = {int(value) for value in np.unique(test_labels) if int(value) > 0}
    combined = train_ids | test_ids
    if not combined:
        raise ValueError("Train/test label maps contain no positive class labels.")
    expected_ids = set(range(1, max(combined) + 1))
    if bool(schema["require_contiguous"]) and combined != expected_ids:
        raise ValueError(
            f"Positive class ids must be contiguous 1..K; found {sorted(combined)}. "
            "Use label_values.replace in the dataset JSON to remap them."
        )
    if bool(schema["require_all_classes_in_train"]) and train_ids != combined:
        raise ValueError(
            "Training labels do not contain every declared class; missing "
            f"{sorted(combined - train_ids)}."
        )
    if bool(schema["require_all_classes_in_test"]) and test_ids != combined:
        raise ValueError(
            "Testing labels do not contain every declared class; missing "
            f"{sorted(combined - test_ids)}."
        )


def _validate_single_split_maps(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    ground_truth: Optional[np.ndarray],
) -> None:
    """Validate one-MAT train/test/GT semantics before any model sees labels."""
    if train_labels.shape != test_labels.shape:
        raise ValueError(
            f"Train-label shape {train_labels.shape} differs from test-label "
            f"shape {test_labels.shape}."
        )
    overlap = (train_labels > 0) & (test_labels > 0)
    if np.any(overlap):
        raise ValueError(
            "The configured single split MAT has train/test overlap at "
            f"{int(overlap.sum())} center pixels."
        )
    if ground_truth is None:
        return
    if ground_truth.shape != train_labels.shape:
        raise ValueError(
            f"Ground-truth shape {ground_truth.shape} differs from split maps "
            f"{train_labels.shape}."
        )
    union = np.where(train_labels > 0, train_labels, test_labels)
    if not np.array_equal(union, ground_truth):
        uncovered = int(np.sum((ground_truth > 0) & (union == 0)))
        extra = int(np.sum((ground_truth == 0) & (union > 0)))
        wrong = int(np.sum((ground_truth > 0) & (union > 0) & (ground_truth != union)))
        raise ValueError(
            "The configured single split MAT is inconsistent: train/test union "
            f"does not reproduce gt (uncovered={uncovered}, extra={extra}, "
            f"wrong_class={wrong})."
        )


def load_mat_hsi_lidar(
    config: Mapping[str, object],
    data_root: Path,
    *,
    validation: Optional[Mapping[str, object]] = None,
    split_seed: Optional[int] = None,
) -> DatasetBundle:
    """Load any configured co-registered HSI + raster modality MAT dataset.

    Protocol rule:
    - all train/test/GT maps come from one configured split MAT;
    - validation enabled: read only its train key, then create train-dev/val;
    - validation disabled: use its fixed train and test keys exactly as stored.
    """
    files = dict(config["files"])
    keys = dict(config["keys"])
    label_split = dict(config["label_split"])
    split_keys = dict(label_split["keys"])
    layouts = dict(config["layouts"])
    expected = dict(config["expected"])
    expected_spatial = _expected_spatial_shape(expected)
    resolution = dict(config["resolution"])
    validation_cfg = dict(validation or {})
    validation_enabled = bool(validation_cfg.get("enabled", False))

    store = MatArrayStore(
        data_root,
        allow_implicit_key_fallback=bool(
            resolution["allow_implicit_key_fallback"]
        ),
    )

    train_labels = align_label(
        store.load(
            label_split["file"], split_keys["train"], "train_labels", 2
        ),
        "train_labels",
        str(layouts["labels"]),
        expected_spatial,
    )
    label_values = dict(config["label_values"])
    replacements = dict(label_values["replace"])
    train_labels = _replace_label_values(train_labels, replacements)
    if int(label_values["background"]) != 0:
        raise ValueError(
            "The common patch pipeline requires background=0; use label_values.replace "
            "to map the source background to zero."
        )

    configured_train_labels = train_labels.copy()
    gt = None

    if validation_enabled:
        # Development mode selects only the train variable from the split MAT.
        # MatArrayStore performs a variable-selective read, so test/GT arrays
        # are not brought into memory here.
        strategy = str(validation_cfg.get("strategy", "stratified_random"))
        if strategy != "stratified_random":
            raise ValueError(
                f"Validation strategy {strategy!r} is not implemented in this release; "
                "use stratified_random. The config field is retained for future predefined/spatial splits."
            )
        if split_seed is None:
            raise ValueError("split_seed is required when validation is enabled")
        train_labels, test_labels, val_meta = stratified_split_label_map(
            configured_train_labels,
            float(validation_cfg.get("ratio", 0.15)),
            int(split_seed),
            min_train_per_class=int(validation_cfg.get("min_train_per_class", 1)),
            min_val_per_class=int(validation_cfg.get("min_val_per_class", 1)),
        )
        _validate_label_schema(
            train_labels,
            test_labels,
            dict(config["label_schema"]),
        )
        split_metadata = {
            "evaluation_split": "validation",
            "label_protocol": "single_mat_fixed_split",
            "configured_train_samples": int((configured_train_labels > 0).sum()),
            "configured_test_samples": None,
            "configured_test_labels_loaded": False,
            "ground_truth_loaded": False,
            **val_meta,
        }
    else:
        # Final mode uses the fixed maps stored in the one configured split MAT.
        test_labels = align_label(
            store.load(
                label_split["file"], split_keys["test"], "test_labels", 2
            ),
            "test_labels",
            str(layouts["labels"]),
            expected_spatial,
        )
        test_labels = _replace_label_values(test_labels, replacements)
        gt_key = split_keys.get("ground_truth")
        if gt_key:
            gt_array = store.load(
                label_split["file"],
                gt_key,
                "ground_truth",
                2,
                required=bool(label_split.get("require_ground_truth", True)),
            )
            if gt_array is not None:
                gt = align_label(
                    gt_array,
                    "ground_truth",
                    str(layouts["labels"]),
                    expected_spatial,
                )
                gt = _replace_label_values(gt, replacements)
        _validate_single_split_maps(train_labels, test_labels, gt)
        _validate_label_schema(
            train_labels,
            test_labels,
            dict(config["label_schema"]),
        )
        split_metadata = {
            "evaluation_split": "configured_test",
            "label_protocol": "single_mat_fixed_split",
            "configured_train_samples": int((train_labels > 0).sum()),
            "configured_test_samples": int((test_labels > 0).sum()),
            "configured_test_labels_loaded": True,
            "ground_truth_loaded": gt is not None,
        }

    split_metadata["label_split_file"] = store.used_files["train_labels"]
    split_metadata["train_key"] = store.used_keys["train_labels"]
    split_metadata["test_key"] = (
        None if validation_enabled else store.used_keys["test_labels"]
    )
    split_metadata["ground_truth_key"] = (
        None if gt is None else store.used_keys.get("ground_truth")
    )

    height, width = train_labels.shape
    hsi = align_cube(
        store.load(files["hsi"], keys["hsi"], "hsi", 3),
        height,
        width,
        "hsi",
        str(layouts["hsi"]),
    )
    hsi = apply_band_selection(hsi, dict(config["band_selection"]))
    lidar = align_cube(
        store.load(files["lidar"], keys["lidar"], "lidar", None),
        height,
        width,
        "lidar",
        str(layouts["lidar"]),
    )

    normalization = dict(config["normalization"])
    if str(normalization["hsi"]) != "train_percentile_minmax":
        raise ValueError("The current degradation protocol requires hsi=train_percentile_minmax.")
    if str(normalization["statistics_source"]) != "train_label_pixels":
        raise ValueError("Normalization statistics_source must be train_label_pixels.")
    if str(normalization["lidar"]) != "train_zscore":
        raise ValueError("The common loader currently supports lidar=train_zscore.")
    if list(normalization["hsi_output_range"]) != [0.0, 1.0]:
        raise ValueError("The current degradation protocol requires hsi_output_range=[0,1].")
    hsi, lidar, norm_stats = normalize_scene(
        hsi,
        lidar,
        train_labels,
        float(normalization["hsi_percentile_low"]),
        float(normalization["hsi_percentile_high"]),
    )

    max_label = int(
        max(
            int(train_labels.max()),
            int(test_labels.max()),
            0 if gt is None else int(gt.max()),
        )
    )
    if max_label <= 1:
        raise ValueError(f"Detected only {max_label} classes. Check MAT keys and labels.")
    bundle = DatasetBundle(
        hsi=hsi,
        lidar=lidar,
        train_labels=train_labels,
        test_labels=test_labels,
        gt=gt,
        num_bands=hsi.shape[2],
        lidar_channels=lidar.shape[2],
        num_classes=max_label,
        norm_stats=norm_stats,
        dataset_name=str(config["dataset"]),
        class_names=tuple(str(value) for value in config["class_names"]),
        metadata={
            "data_root": str(data_root.resolve()),
            "loader": str(config["loader"]),
            "files": store.used_files,
            "keys": store.used_keys,
            "split": split_metadata,
            "normalization_statistics_source": (
                "train_dev_label_pixels" if validation_enabled else "configured_train_label_pixels"
            ),
        },
    )
    _validate_expected(bundle, expected)
    if bundle.class_names and len(bundle.class_names) != bundle.num_classes:
        raise ValueError(
            f"Configured {len(bundle.class_names)} class names for "
            f"{bundle.num_classes} loaded classes."
        )
    print(
        f"[data] {bundle.dataset_name} loaded: HSI={hsi.shape}, LiDAR={lidar.shape}, "
        f"train={int((train_labels > 0).sum())}, "
        f"{split_metadata['evaluation_split']}={int((test_labels > 0).sum())}, classes={max_label}"
    )
    if validation_enabled:
        print("[data] configured test/GT variables: SEALED / NOT LOADED")
    return bundle


__all__ = [
    "MatArrayStore",
    "align_cube",
    "align_label",
    "apply_band_selection",
    "load_mat_hsi_lidar",
    "normalize_scene",
    "stratified_split_label_map",
]
