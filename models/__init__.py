"""Public model API for PCRFusion."""

from .clean_fusion import CleanFusionBackbone
from .pcrfusion import (
    PCRFusion,
    PrototypeGuidedRepairBranch,
    ModalityDegradationEncoder,
    HSIMultiBranchDegradationEncoder,
    DegradationPrototypeBank,
    ResidualRepairExpert,
    ExpertConditionedRepairUtility,
    load_clean_fusion_from_checkpoint,
    count_parameters,
)

__all__ = [
    "PCRFusion",
    "CleanFusionBackbone",
    "PrototypeGuidedRepairBranch",
    "ModalityDegradationEncoder",
    "HSIMultiBranchDegradationEncoder",
    "DegradationPrototypeBank",
    "ResidualRepairExpert",
    "ExpertConditionedRepairUtility",
    "load_clean_fusion_from_checkpoint",
    "count_parameters",
]
