"""Internal random-stream derivation from the single configured seed.

Users must not edit this file to change experiment seeds. Public split/model/
corruption seeds are configured in ``configs/experiments/*.json``.
"""

from __future__ import annotations

from types import MappingProxyType
import hashlib


MAX_SEED = 2_147_483_647
_FOLD_STRIDE = 100_003

# Fixed internal stream separation.  These are implementation constants, not
# independently tunable hyperparameters.  The offsets preserve the validated established random streams for reproducibility.
_STREAM_OFFSETS = MappingProxyType(
    {
        "data": 0,
        "teacher": 0,
        "corruption": 0,
        "worker": 0,
        "dataloader": 1_009,
        "degradation_pretrain": 2_003,
        "reserved_warmup": 3_001,
        "reserved_joint_full": 4_001,
        "reserved_joint_inner": 5_003,
        "reserved_condition_shuffle": 6_007,
        "bootstrap": 59_656,
        "expert": 63_001,
        "router": 63_017,
        "clean_fusion_loader": 998_046_825,
        "prototype_refresh_hsi": 594_068_467,
        "prototype_refresh_lidar": 790_682_787,
        "joint_training_hsi": 895_368_741,
        "joint_training_lidar": 657_644_425,
        "prototype_refresh_epoch": 917_485_222,
        "robust_provider": 408_632_629,
        "robust_fusion_loader": 98_681_056,
    }
)


def validate_seed(seed: int) -> int:
    value = int(seed)
    if not 0 <= value <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {value}.")
    return value


def derived_seed(seed: int, stream: str, index: int = 0) -> int:
    """Derive one deterministic internal stream from the configured seed."""
    base = validate_seed(seed)
    if stream in _STREAM_OFFSETS:
        offset = int(_STREAM_OFFSETS[stream])
    else:
        # Named internal streams are deterministically separated; they are not additional tunable seeds.
        digest = hashlib.sha1(str(stream).encode("utf-8")).hexdigest()[:8]
        offset = 100_000 + (int(digest, 16) % 1_000_000_000)
    return (base + offset + int(index)) % MAX_SEED


def fold_seed(seed: int, fold_index: int) -> int:
    """Derive a stable seed for a zero-based outer LODO fold."""
    index = int(fold_index)
    if index < 0:
        raise ValueError("fold_index must be non-negative.")
    return validate_seed(seed) + (index + 1) * _FOLD_STRIDE


def offset_seed(seed: int, offset: int) -> int:
    """Derive a deterministic diagnostic stream without exposing another setting."""
    return validate_seed(seed) + int(offset)


__all__ = ["derived_seed", "fold_seed", "offset_seed", "validate_seed"]
