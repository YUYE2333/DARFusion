from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from degradation.protocol import DualModalDegradationProtocol
from degradation.provider import DualModalWholeSceneProvider
from datasets.registry import load_dataset
from datasets.common import make_loaders, reset_loader_random_streams, seed_everything
from datasets.spec import DatasetSpec
from random_control import derived_seed, validate_seed
from utils.config import load_experiment
from models.pcrfusion import PCRFusion, load_clean_fusion_from_checkpoint, count_parameters
from utils.diagnostics import write_rows, write_matrix
from repair.low_mix import additional_loss, ramp
from utils.data_signature import signature, provider_cache, code_signature, evaluation_loader
import hashlib
from utils.resume import ResumeStore, capture_random, restore_random, atomic_torch, EvaluationJournal, state_hash, safe_load


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent / "configs" / "experiments" / "houston2013.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PCRFusion Stage-II Robust Fusion Learning")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--clean-fusion-checkpoint", type=Path, required=True,
                   help="Path to the Stage-I Clean Fusion best.pt checkpoint.")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--split-file", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument("--batch-size", type=int, default=None, help="Override config batch size")
    p.add_argument("--device", default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume epochs and completed evaluation conditions")
    p.add_argument("--skip-final-evaluation", action="store_true", help="Save the trained checkpoint without final evaluation")
    p.add_argument("--eval-profile", choices=("core", "full"), default="full",
                   help="core=16 conditions; full=complete condition matrix")
    return p.parse_args()


def load_bundle(
    config_path: Path,
    data_root: Path | None,
    split_file: Path | None,
    seed_override: int | None,
    batch_override: int | None,
    device_override: str | None,
):
    resolved = load_experiment(
        config_path,
        data_root_override=data_root,
        split_file_override=split_file,
        model_seed_override=seed_override,
        batch_size_override=batch_override,
        device_override=device_override,
    )
    main = resolved["config"]
    resolved["robust_fusion_runtime"] = {
        "experiment": resolved["experiment"],
        "runtime": main["runtime"],
        "model": main["robust_fusion"]["model"],
        "training": main["robust_fusion"]["training"],
        "evaluation": main["evaluation"],
        "gates": main.get("gates", {}),
        "output_root": resolved["paths"]["output_root"],
    }
    return resolved


def get_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        print("[runtime] CUDA unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(name)


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_scaler(device: torch.device, enabled: bool):
    active = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=active)
    # Compatibility with the older PyTorch releases still used on several
    # remote-sensing training workstations.
    return torch.cuda.amp.GradScaler(enabled=active)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> Dict[str, object]:
    labels = np.asarray(labels, np.int64); predictions = np.asarray(predictions, np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(labels, predictions):
        if 0 <= y < num_classes and 0 <= p < num_classes:
            cm[y, p] += 1
    oa = float(np.mean(labels == predictions))
    recalls = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(num_classes, dtype=float), where=cm.sum(1) > 0)
    valid = cm.sum(1) > 0
    aa = float(recalls[valid].mean()) if valid.any() else float("nan")
    total = int(cm.sum())
    pe = float((cm.sum(0) * cm.sum(1)).sum()) / float(max(total * total, 1))
    kappa = float((oa - pe) / max(1.0 - pe, 1e-12))
    return {
        "OA": oa, "AA": aa, "Kappa": kappa, "EachAcc": recalls.tolist(),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate_clean_fusion(teacher, loader, device: torch.device, num_classes: int) -> Dict[str, object]:
    teacher.eval(); ys = []; ps = []
    for batch in loader:
        out = teacher(batch["hsi"].to(device), batch["lidar"].to(device))
        ys.append(batch["label"].numpy()); ps.append(out["logits"].argmax(1).cpu().numpy())
    return classification_metrics(np.concatenate(ys), np.concatenate(ps), num_classes)

def _token_diversity(tokens: Tensor, margin: float) -> Tensor:
    if tokens.shape[1] <= 1:
        return tokens.new_zeros(())
    sim = tokens @ tokens.transpose(1, 2)
    eye = torch.eye(tokens.shape[1], device=tokens.device, dtype=torch.bool)[None]
    vals = sim.masked_select(~eye.expand_as(sim))
    return F.relu(vals - float(margin)).mean()


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)


def _balanced_condition_schedule(
    total_samples: int,
    known_names: Sequence[str],
    severities: Sequence[int],
    rng: np.random.Generator,
    *,
    clean_ratio: float = 0.0,
    include_clean: bool = False,
) -> List[Tuple[int, str | None, int]]:
    """Build one epoch of sample-level conditions with exact dataset-level balancing.

    Batch size only changes how this schedule is chunked into minibatches; it does not
    change the number of clean/family/severity samples exposed in an epoch.
    """
    n = max(0, int(total_samples))
    sev = [int(x) for x in severities]
    degraded = [(int(tid), str(name), int(s)) for tid, name in enumerate(known_names) for s in sev]
    if n == 0:
        return []
    if not degraded and not include_clean:
        raise ValueError("No degradation conditions available for scheduling.")

    n_clean = int(round(n * float(clean_ratio))) if include_clean else 0
    n_clean = min(max(n_clean, 0), n)
    n_degraded = n - n_clean
    schedule: List[Tuple[int, str | None, int]] = []

    if include_clean and n_clean:
        clean_tid = len(known_names)
        schedule.extend([(clean_tid, None, 0)] * n_clean)

    if n_degraded and degraded:
        k = len(degraded)
        base, rem = divmod(n_degraded, k)
        order = rng.permutation(k).tolist()
        counts = [base] * k
        for idx in order[:rem]:
            counts[int(idx)] += 1
        for cond, count in zip(degraded, counts):
            schedule.extend([cond] * int(count))

    rng.shuffle(schedule)
    return schedule


def _extract_mixed_modality(
    provider, rows: Tensor, cols: Tensor, device: torch.device, *, modality: str,
    conditions: Sequence[Tuple[int, str | None, int]],
) -> Tensor:
    """Extract a mixed-condition minibatch while preserving the existing cached-scene corruption protocol."""
    if len(conditions) != int(rows.numel()):
        raise ValueError("Condition count must match minibatch size.")
    rr = rows.detach().cpu().numpy().astype(np.int64)
    cc = cols.detach().cpu().numpy().astype(np.int64)
    groups: Dict[Tuple[str | None, int], List[int]] = {}
    for i, (_, name, severity) in enumerate(conditions):
        groups.setdefault((name, int(severity)), []).append(i)

    out = None
    for (name, severity), idxs in groups.items():
        if modality == "hsi":
            scene = provider.get_hsi_scene(name, severity if name is not None else 0)
        elif modality == "lidar":
            scene = provider.get_lidar_scene(name, severity if name is not None else 0)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        arr = np.stack([provider._extract(scene, int(rr[i]), int(cc[i])) for i in idxs])
        if out is None:
            out = np.empty((len(conditions), *arr.shape[1:]), dtype=np.float32)
        out[np.asarray(idxs, dtype=np.int64)] = arr
    if out is None:
        raise RuntimeError("Empty mixed-condition minibatch.")
    return torch.from_numpy(np.ascontiguousarray(out)).to(device, non_blocking=True)


def _hsi_backbone(encoder: nn.Module) -> nn.Module:
    return getattr(encoder, "backbone", encoder)


def configure_trainable_groups(model: PCRFusion, cfg: Mapping[str, object]):
    """Configure partial-unfreezing groups; PCA and early layers stay frozen."""
    freeze_all(model)
    repair_modules = [model.hsi_repair_branch.degradation_encoder, model.lidar_repair_branch.degradation_encoder]
    repair_modules += list(model.hsi_repair_branch.experts) + list(model.lidar_repair_branch.experts)
    for module in repair_modules:
        unfreeze(module)

    partial_cfg = dict(cfg.get("partial_unfreeze", {}))
    backbone_modules: List[nn.Module] = []
    fusion_modules: List[nn.Module] = []
    if bool(partial_cfg.get("enabled", True)):
        h_backbone = _hsi_backbone(model.hsi_encoder)
        for name in partial_cfg.get("hsi_modules", ("down2", "stage3")):
            if not hasattr(h_backbone, str(name)):
                raise AttributeError(f"HSI backbone has no configured unfreeze module {name!r}")
            backbone_modules.append(getattr(h_backbone, str(name)))
        for name in partial_cfg.get("lidar_modules", ("down2", "stage3")):
            if not hasattr(model.lidar_encoder, str(name)):
                raise AttributeError(f"LiDAR backbone has no configured unfreeze module {name!r}")
            backbone_modules.append(getattr(model.lidar_encoder, str(name)))
        if bool(partial_cfg.get("fusion", True)):
            fusion_modules.extend([model.fusion, model.pool_f])
        if bool(partial_cfg.get("classifier", True)):
            fusion_modules.append(model.classifier)
        for module in backbone_modules + fusion_modules:
            unfreeze(module)

    def unique_params(modules: Sequence[nn.Module]) -> List[nn.Parameter]:
        seen = set(); result = []
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter)); result.append(parameter)
        return result

    groups = []
    repair_params = unique_params(repair_modules)
    backbone_params = unique_params(backbone_modules)
    fusion_params = unique_params(fusion_modules)
    if repair_params:
        groups.append({"params": repair_params, "lr": float(cfg.get("learning_rate", 8e-4)), "name": "repair_branch"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(cfg.get("backbone_learning_rate", 4e-5)), "name": "backbone_tail"})
    if fusion_params:
        groups.append({"params": fusion_params, "lr": float(cfg.get("fusion_learning_rate", 1.2e-4)), "name": "fusion_head"})
    modules = repair_modules + backbone_modules + fusion_modules
    return groups, modules


def _hard_feature(branch, base: Tensor, conditions):
    candidates = torch.stack([expert(base)["feature"] for expert in branch.experts], dim=1)
    type_ids = torch.tensor([int(c[0]) for c in conditions], dtype=torch.long, device=base.device)
    is_clean = type_ids >= len(branch.experts)
    safe = type_ids.clamp(0, max(len(branch.experts) - 1, 0))
    hard = candidates[torch.arange(base.shape[0], device=base.device), safe]
    hard = torch.where(is_clean[:, None, None, None], base, hard)
    return hard, candidates, is_clean


@torch.no_grad()
def rebuild_prototypes(
    model: PCRFusion, loader, provider, *, hsi_known_names: Sequence[str],
    lidar_known_names: Sequence[str], device: torch.device, seed: int,
    clean_ratio: float,
) -> Dict[str, int]:
    """Recompute all prototype statistics from one fresh, post-update pass."""
    model.eval()
    severities = tuple(int(x) for x in provider.protocol.severity_levels)
    rng_h = np.random.default_rng(derived_seed(seed, "prototype_refresh_hsi"))
    rng_l = np.random.default_rng(derived_seed(seed, "prototype_refresh_lidar"))
    hs = _balanced_condition_schedule(len(loader.dataset), hsi_known_names, severities, rng_h, clean_ratio=clean_ratio, include_clean=True)
    ls = _balanced_condition_schedule(len(loader.dataset), lidar_known_names, severities, rng_l, clean_ratio=clean_ratio, include_clean=True)
    h_storage: Dict[Tuple[int, int], List[Tensor]] = {}; l_storage: Dict[Tuple[int, int], List[Tensor]] = {}
    cursor = 0
    for batch in loader:
        size = int(batch["label"].shape[0]); hc = hs[cursor:cursor + size]; lc = ls[cursor:cursor + size]; cursor += size
        h = _extract_mixed_modality(provider, batch["row"], batch["col"], device, modality="hsi", conditions=hc)
        l = _extract_mixed_modality(provider, batch["row"], batch["col"], device, modality="lidar", conditions=lc)
        ht = model.hsi_repair_branch.degradation_encoder(h)["tokens"].detach().cpu()
        lt = model.lidar_repair_branch.degradation_encoder(l)["tokens"].detach().cpu()
        for tokens, conditions, storage in ((ht, hc, h_storage), (lt, lc, l_storage)):
            for i, (tid, name, severity) in enumerate(conditions):
                key = (int(tid), 0 if name is None else int(severity))
                storage.setdefault(key, []).append(tokens[i:i + 1])

    def fit(branch, storage, known_count: int) -> None:
        branch.prototype_bank.reset()
        for (type_id, severity), pieces in storage.items():
            branch.prototype_bank.fit_condition(torch.cat(pieces, dim=0).to(device), type_id, severity)
        for type_id in range(known_count):
            if not bool(branch.prototype_bank.initialized[type_id].any()):
                raise RuntimeError(f"Fresh prototype pass missed degradation family {type_id}")
        if not bool(branch.prototype_bank.initialized[branch.clean_type_id, 0].all()):
            raise RuntimeError("Fresh prototype pass missed the clean condition")

    fit(model.hsi_repair_branch, h_storage, len(hsi_known_names))
    fit(model.lidar_repair_branch, l_storage, len(lidar_known_names))
    return {"hsi_conditions": len(h_storage), "lidar_conditions": len(l_storage)}


def train_robust_fusion_stage(
    model: PCRFusion, teacher, loader, selection_loader, provider, *,
    hsi_known_names: Sequence[str], lidar_known_names: Sequence[str],
    device: torch.device, cfg: Mapping[str, object], weight_decay: float,
    grad_clip: float, amp: bool, seed: int, num_classes: int,
    history_dir: Path | None = None,
    resume: bool = False, resume_identity: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Robust fusion training with partial unfreezing, clean anchoring, and hard routing supervision."""
    model.set_repairability_ready(False)
    parameter_groups, train_modules = configure_trainable_groups(model, cfg)
    params = [p for group in parameter_groups for p in group["params"]]
    if not params:
        raise RuntimeError("PCRFusion has no trainable parameters")
    opt = torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)
    scaler = make_scaler(device, amp)
    rng_h = np.random.default_rng(derived_seed(seed, "joint_training_hsi"))
    rng_l = np.random.default_rng(derived_seed(seed, "joint_training_lidar"))
    severities = tuple(int(x) for x in provider.protocol.severity_levels)
    clean_ratio = float(cfg.get("clean_sample_ratio", 0.20))
    epochs = int(cfg.get("epochs", 50))
    refresh_cfg = dict(cfg.get("prototype_refresh", {}))
    selection_every = max(1, int(cfg.get("selection_every_epochs", 5)))
    selection_score_cfg = dict(cfg.get("validation_score", {}))
    history: List[Dict[str, object]] = []
    best_state = None; best_metrics = None; best_epoch = None; best_score = -float("inf")

    teacher.eval(); model.eval()
    composition_initialized = False
    if model.composition_config["enabled"] and not bool(refresh_cfg.get("enabled", True)):
        raise ValueError("Composition training requires prototype refresh enabled")
    print("[trainable] " + ", ".join(f"{g['name']}={sum(p.numel() for p in g['params']):,}@{g['lr']:.2e}" for g in parameter_groups))
    store = ResumeStore(history_dir,dict(stage='robust',cfg=dict(cfg),seed=seed,teacher=state_hash(teacher.state_dict()),identity=resume_identity),resume) if history_dir is not None else None
    recovered=store.load() if store is not None else None
    start_epoch=1
    if recovered is not None:
        model.load_state_dict(recovered['model']);opt.load_state_dict(recovered['optimizer']);scaler.load_state_dict(recovered['scaler'])
        history=recovered['history'];best_state=recovered['best_state'];best_metrics=recovered['best_metrics'];best_epoch=recovered['best_epoch'];best_score=recovered['best_score']
        composition_initialized=recovered['composition_initialized']
        rng_h.bit_generator.state=recovered['rng_h'];rng_l.bit_generator.state=recovered['rng_l']
        restore_random(recovered['random'],dict(train=loader,selection=selection_loader))
        start_epoch=int(recovered['epoch'])+1
        print(f"[resume:StageII] next epoch={start_epoch}/{epochs}",flush=True)
    for epoch in range(start_epoch, epochs + 1):
        composition_active = model.composition_config["enabled"] and ramp(epoch, epochs, model.composition_config) > 0
        if composition_active and not composition_initialized:
            generator = getattr(loader, "generator", None)
            state = generator.get_state() if generator is not None else None
            try:
                rebuild_prototypes(model, loader, provider, hsi_known_names=hsi_known_names,
                                   lidar_known_names=lidar_known_names, device=device,
                                   seed=derived_seed(seed, "composition_initial_prototypes", epoch), clean_ratio=clean_ratio)
            finally:
                if generator is not None: generator.set_state(state)
            composition_initialized = True
        model.eval()
        for module in train_modules:
            module.train()
        h_schedule = _balanced_condition_schedule(len(loader.dataset), hsi_known_names, severities, rng_h, clean_ratio=clean_ratio, include_clean=True)
        l_schedule = _balanced_condition_schedule(len(loader.dataset), lidar_known_names, severities, rng_l, clean_ratio=clean_ratio, include_clean=True)
        cursor = 0; total = correct = 0
        keys = ("loss", "deg", "feature", "semantic", "classification", "identity", "residual", "clean_anchor", "clean_kd")
        keys += ("composition_feature", "composition_semantic", "composition_classification", "composition_route", "composition_ramp")
        sums = {key: 0.0 for key in keys}
        for batch in loader:
            size = int(batch["label"].shape[0]); hc = h_schedule[cursor:cursor + size]; lc = l_schedule[cursor:cursor + size]; cursor += size
            deg_h = _extract_mixed_modality(provider, batch["row"], batch["col"], device, modality="hsi", conditions=hc)
            deg_l = _extract_mixed_modality(provider, batch["row"], batch["col"], device, modality="lidar", conditions=lc)
            clean_h = batch["hsi"].to(device); clean_l = batch["lidar"].to(device); labels = batch["label"].to(device)
            h_tid = torch.tensor([int(c[0]) for c in hc], dtype=torch.long, device=device)
            l_tid = torch.tensor([int(c[0]) for c in lc], dtype=torch.long, device=device)
            h_sev = torch.tensor([0.0 if c[1] is None else float(c[2]) / max(severities) for c in hc], dtype=torch.float32, device=device)
            l_sev = torch.tensor([0.0 if c[1] is None else float(c[2]) / max(severities) for c in lc], dtype=torch.float32, device=device)
            with torch.no_grad():
                teacher_out = teacher(clean_h, clean_l)
                target_h = teacher.hsi_encoder(clean_h); target_l = teacher.lidar_encoder(clean_l)

            opt.zero_grad(set_to_none=True)
            with amp_context(device, amp):
                h_base = model.hsi_encoder(deg_h); l_base = model.lidar_encoder(deg_l)
                h_deg = model.hsi_repair_branch.degradation_encoder(deg_h); l_deg = model.lidar_repair_branch.degradation_encoder(deg_l)
                h_type = F.cross_entropy(h_deg["type_logits"], h_tid); l_type = F.cross_entropy(l_deg["type_logits"], l_tid)
                h_sv = F.smooth_l1_loss(h_deg["severity"], h_sev); l_sv = F.smooth_l1_loss(l_deg["severity"], l_sev)
                h_div = _token_diversity(h_deg["tokens"], float(cfg.get("token_diversity_margin", 0.85)))
                l_div = _token_diversity(l_deg["tokens"], float(cfg.get("token_diversity_margin", 0.85)))
                deg_loss = 0.5 * (h_type + l_type) + float(cfg.get("severity_loss_weight", 0.5)) * 0.5 * (h_sv + l_sv) + float(cfg.get("token_diversity_weight", 0.02)) * 0.5 * (h_div + l_div)
                h_rep, h_candidates, h_clean_mask = _hard_feature(model.hsi_repair_branch, h_base, hc)
                l_rep, l_candidates, l_clean_mask = _hard_feature(model.lidar_repair_branch, l_base, lc)
                feature_loss = 0.5 * (F.smooth_l1_loss(h_rep, target_h) + F.smooth_l1_loss(l_rep, target_l))
                fused = model.classify_features(h_rep, l_rep)
                semantic_loss = F.smooth_l1_loss(fused["semantic"], teacher_out["semantic"].detach())
                cls_loss = F.cross_entropy(fused["logits"], labels)
                identity_terms = []
                if bool(h_clean_mask.any()): identity_terms.append(F.smooth_l1_loss(h_candidates[h_clean_mask], target_h[h_clean_mask, None].expand_as(h_candidates[h_clean_mask])))
                if bool(l_clean_mask.any()): identity_terms.append(F.smooth_l1_loss(l_candidates[l_clean_mask], target_l[l_clean_mask, None].expand_as(l_candidates[l_clean_mask])))
                identity_loss = torch.stack(identity_terms).mean() if identity_terms else cls_loss.new_zeros(())
                residual_loss = 0.5 * ((h_candidates - h_base[:, None]).abs().mean() + (l_candidates - l_base[:, None]).abs().mean())
                clean_h_student = model.hsi_encoder(clean_h); clean_l_student = model.lidar_encoder(clean_l)
                clean_out = model.classify_features(clean_h_student, clean_l_student)
                clean_anchor = F.cross_entropy(clean_out["logits"], labels) + 0.5 * (F.smooth_l1_loss(clean_h_student, target_h) + F.smooth_l1_loss(clean_l_student, target_l))
                clean_kd = _kl_logits(clean_out["logits"], teacher_out["logits"])
                loss = (float(cfg.get("degradation_loss_weight", 1.0)) * deg_loss +
                        float(cfg.get("feature_loss_weight", 1.0)) * feature_loss +
                        float(cfg.get("semantic_loss_weight", 0.35)) * semantic_loss +
                        float(cfg.get("classification_loss_weight", 0.25)) * cls_loss +
                        float(cfg.get("clean_identity_weight", 0.25)) * identity_loss +
                        float(cfg.get("residual_weight", 0.02)) * residual_loss +
                        float(cfg.get("clean_anchor_weight", 0.5)) * clean_anchor +
                        float(cfg.get("clean_logit_distillation_weight", 0.2)) * clean_kd)
                extra, composition_logs = additional_loss(
                    model, epoch, epochs, h_deg, l_deg, h_base, l_base, h_candidates, l_candidates,
                    target_h, target_l, teacher_out, labels, h_clean_mask, l_clean_mask)
                loss = loss + extra
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(params, grad_clip); scaler.step(opt); scaler.update()
            n = labels.numel(); total += n; correct += int((fused["logits"].argmax(1) == labels).sum())
            values = {"loss": loss, "deg": deg_loss, "feature": feature_loss, "semantic": semantic_loss, "classification": cls_loss,
                      "identity": identity_loss, "residual": residual_loss, "clean_anchor": clean_anchor, "clean_kd": clean_kd, **composition_logs}
            for key, value in values.items(): sums[key] += float(value.detach()) * n

        is_selection_epoch = epoch == 1 or epoch == epochs or epoch % selection_every == 0
        # Hard-route training does not consume prototypes.  Rebuild them only
        # when they are actually needed for checkpoint selection/final inference.
        should_refresh = bool(refresh_cfg.get("enabled", True)) and (is_selection_epoch or composition_active)
        if should_refresh:
            loader_generator = getattr(loader, "generator", None)
            generator_state = loader_generator.get_state() if loader_generator is not None else None
            try:
                refresh = rebuild_prototypes(
                    model, loader, provider, hsi_known_names=hsi_known_names, lidar_known_names=lidar_known_names,
                    device=device, seed=derived_seed(seed, "prototype_refresh_epoch", epoch),
                    clean_ratio=float(refresh_cfg.get("clean_sample_ratio", clean_ratio)),
                )
            finally:
                if loader_generator is not None and generator_state is not None:
                    loader_generator.set_state(generator_state)
        else:
            refresh = {"hsi_conditions": 0, "lidar_conditions": 0}

        row = {"epoch": epoch, "train_OA": correct / max(total, 1),
               **refresh, **{key: value / max(total, 1) for key, value in sums.items()}}
        if is_selection_epoch:
            selection = evaluate_known_validation(
                model, selection_loader, provider, device=device, num_classes=num_classes,
                hsi_known_names=hsi_known_names, lidar_known_names=lidar_known_names,
                severity_levels=severities, score_cfg=selection_score_cfg, route="cosine_experts",
            )
            row.update({f"selection_{key}": value for key, value in selection.items()})
            if float(selection["score"]) > best_score:
                best_score = float(selection["score"]); best_metrics = copy.deepcopy(selection)
                best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch
        history.append(row)
        if store is not None:
            store.save(dict(epoch=epoch,goal_epochs=epochs,model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                            history=history,best_state=best_state,best_metrics=best_metrics,best_epoch=best_epoch,best_score=best_score,
                            composition_initialized=composition_initialized,rng_h=rng_h.bit_generator.state,rng_l=rng_l.bit_generator.state,
                            random=capture_random(dict(train=loader,selection=selection_loader))))
        if history_dir is not None:
            write_rows(history_dir / "pcrfusion_history.csv", history)
        selected = " *best" if best_epoch == epoch else ""
        score_text = f" select={row['selection_score']:.5f}" if "selection_score" in row else ""
        print(f"[robust-fusion] epoch={epoch:03d} loss={row['loss']:.5f} train_OA={row['train_OA']:.4f}{score_text}{selected}")

    if best_state is None:
        raise RuntimeError(" robust fusion training did not select a checkpoint")
    model.load_state_dict(best_state); freeze_all(model)
    return {"best_epoch": best_epoch, "best_selection": best_metrics, "best_score": best_score, "history": history}

def _kl_logits(student: Tensor, teacher: Tensor, temperature: float = 2.0) -> Tensor:
    t = float(temperature)
    return F.kl_div(F.log_softmax(student / t, dim=1), F.softmax(teacher.detach() / t, dim=1), reduction="batchmean") * (t * t)


@torch.no_grad()
def evaluate_condition(model: PCRFusion, loader, provider, *, device: torch.device, num_classes: int,
                       hsi_name=None, hsi_severity=0, lidar_name=None, lidar_severity=0, cross_name=None, cross_severity=0,
                       route: str = "auto") -> Dict[str, object]:
    model.eval(); ys = []; ps = []; ch = []; cl = []; rh = []; rl = []; ih = []; il = []; uh = []; ul = []; ah = []; al = []; conf = []; ent = []; hw = []; lw = []
    for batch in loader:
        h, l = provider.extract_condition(batch["row"], batch["col"], device, hsi_name, hsi_severity, lidar_name, lidar_severity, cross_name, cross_severity)
        out = model(h, l, route=route)
        prob = F.softmax(out["logits"], dim=1)
        pred = prob.argmax(1)
        ys.append(batch["label"].numpy()); ps.append(pred.cpu().numpy())
        conf.append(prob.max(1).values.cpu().numpy())
        ent.append((-(prob.clamp_min(1e-12) * prob.clamp_min(1e-12).log()).sum(1)).cpu().numpy())
        ch.append(out["hsi_clean_probability"].cpu().numpy()); cl.append(out["lidar_clean_probability"].cpu().numpy())
        rh.append(out["hsi_repair_mass"].cpu().numpy()); rl.append(out["lidar_repair_mass"].cpu().numpy())
        ih.append(out["hsi_identity_mass"].cpu().numpy()); il.append(out["lidar_identity_mass"].cpu().numpy())
        uh.append(out["hsi_selected_repair_utility"].cpu().numpy()); ul.append(out["lidar_selected_repair_utility"].cpu().numpy())
        ah.append((out["hsi_selected_repair_utility"] > model.hsi_repair_branch.repair_activation_threshold).cpu().numpy())
        al.append((out["lidar_selected_repair_utility"] > model.lidar_repair_branch.repair_activation_threshold).cpu().numpy())
        if out["hsi_weights"].shape[1] > 0:
            hw.append(out["hsi_weights"].cpu().numpy())
        if out["lidar_weights"].shape[1] > 0:
            lw.append(out["lidar_weights"].cpu().numpy())
    yy = np.concatenate(ys); pp = np.concatenate(ps); cc = np.concatenate(conf); ee = np.concatenate(ent)
    m = classification_metrics(yy, pp, num_classes)
    correct_mask = yy == pp
    hweights_arr = np.concatenate(hw) if hw else None
    lweights_arr = np.concatenate(lw) if lw else None
    def _family_stats(arr):
        if arr is None or arr.size == 0:
            return float("nan"), float("nan")
        safe = np.clip(arr, 1e-12, 1.0)
        entropy = -(safe * np.log(safe)).sum(axis=1)
        top1 = arr.max(axis=1)
        return float(entropy.mean()), float(top1.mean())
    h_family_entropy, h_family_top1 = _family_stats(hweights_arr)
    l_family_entropy, l_family_top1 = _family_stats(lweights_arr)
    m.update({
        "mean_hsi_clean_probability": float(np.concatenate(ch).mean()),
        "mean_lidar_clean_probability": float(np.concatenate(cl).mean()),
        "mean_hsi_repair_mass": float(np.concatenate(rh).mean()),
        "mean_lidar_repair_mass": float(np.concatenate(rl).mean()),
        "mean_hsi_identity_mass": float(np.concatenate(ih).mean()),
        "mean_lidar_identity_mass": float(np.concatenate(il).mean()),
        "mean_hsi_selected_repair_utility": float(np.concatenate(uh).mean()),
        "mean_lidar_selected_repair_utility": float(np.concatenate(ul).mean()),
        "hsi_repair_selected_rate": float(np.concatenate(ah).mean()),
        "lidar_repair_selected_rate": float(np.concatenate(al).mean()),
        "mean_prediction_confidence": float(cc.mean()),
        "mean_prediction_entropy": float(ee.mean()),
        "correct_mean_confidence": float(cc[correct_mask].mean()) if np.any(correct_mask) else float("nan"),
        "wrong_mean_confidence": float(cc[~correct_mask].mean()) if np.any(~correct_mask) else float("nan"),
        "mean_hsi_family_weights": hweights_arr.mean(axis=0).tolist() if hweights_arr is not None else [],
        "mean_lidar_family_weights": lweights_arr.mean(axis=0).tolist() if lweights_arr is not None else [],
        "mean_hsi_family_entropy": h_family_entropy,
        "mean_lidar_family_entropy": l_family_entropy,
        "mean_hsi_top1_family_weight": h_family_top1,
        "mean_lidar_top1_family_weight": l_family_top1,
    })
    return m


@torch.no_grad()
def evaluate_known_validation(
    model: PCRFusion, loader, provider, *, device: torch.device, num_classes: int,
    hsi_known_names: Sequence[str], lidar_known_names: Sequence[str],
    severity_levels: Sequence[int], score_cfg: Mapping[str, object], route: str = "auto",
) -> Dict[str, float]:
    """Known-only validation used for prototype-guided routing checkpoint selection.

    Held-out HSI/LiDAR degradation families are deliberately never evaluated here.
    To keep validation inexpensive, all known families are evaluated at the middle
    configured severity (s2 for the standard 1/2/3 protocol).
    """
    levels = [int(x) for x in severity_levels]
    severity = levels[len(levels) // 2]
    clean = evaluate_condition(model, loader, provider, device=device, num_classes=num_classes, route=route)
    hvals = [
        evaluate_condition(model, loader, provider, device=device, num_classes=num_classes, hsi_name=name, hsi_severity=severity, route=route)["OA"]
        for name in hsi_known_names
    ]
    lvals = [
        evaluate_condition(model, loader, provider, device=device, num_classes=num_classes, lidar_name=name, lidar_severity=severity, route=route)["OA"]
        for name in lidar_known_names
    ]
    hmean = float(np.mean(hvals)) if hvals else float(clean["OA"])
    lmean = float(np.mean(lvals)) if lvals else float(clean["OA"])
    wc = float(score_cfg.get("clean_weight", 0.5)); wh = float(score_cfg.get("hsi_known_weight", 0.25)); wl = float(score_cfg.get("lidar_known_weight", 0.25))
    denom = max(wc + wh + wl, 1e-12)
    score = (wc * float(clean["OA"]) + wh * hmean + wl * lmean) / denom
    return {"score": float(score), "clean_OA": float(clean["OA"]), "hsi_known_mOA": hmean, "lidar_known_mOA": lmean, "severity": float(severity)}


def condition_rows(protocol, h_unknown: str, l_unknown: str, ev: Mapping[str, object], quick: bool, uu_matrix: bool = False) -> List[Dict[str, object]]:
    h_all = list(protocol.hsi_degradations); l_all = list(protocol.lidar_degradations)
    h_known = [x for x in h_all if x != h_unknown]; l_known = [x for x in l_all if x != l_unknown]
    rows = [{"condition_group":"clean","condition":"clean_clean","hsi_name":None,"hsi_severity":0,"lidar_name":None,"lidar_severity":0,"cross_name":None,"cross_severity":0}]
    for s in ev["single_severity_levels"]:
        if not quick:
            for h in h_known: rows.append({"condition_group":"hsi_known","condition":f"hsi_known:{h}:s{s}","hsi_name":h,"hsi_severity":int(s),"lidar_name":None,"lidar_severity":0,"cross_name":None,"cross_severity":0})
        rows.append({"condition_group":"hsi_unknown","condition":f"hsi_unknown:{h_unknown}:s{s}","hsi_name":h_unknown,"hsi_severity":int(s),"lidar_name":None,"lidar_severity":0,"cross_name":None,"cross_severity":0})
        if not quick:
            for l in l_known: rows.append({"condition_group":"lidar_known","condition":f"lidar_known:{l}:s{s}","hsi_name":None,"hsi_severity":0,"lidar_name":l,"lidar_severity":int(s),"cross_name":None,"cross_severity":0})
        rows.append({"condition_group":"lidar_unknown","condition":f"lidar_unknown:{l_unknown}:s{s}","hsi_name":None,"hsi_severity":0,"lidar_name":l_unknown,"lidar_severity":int(s),"cross_name":None,"cross_severity":0})
    if uu_matrix:
        uu_levels = [int(s) for s in ev.get("single_severity_levels", [1, 2, 3])]
        for hs in uu_levels:
            for ls in uu_levels:
                rows.append({
                    "condition_group":"dual_unknown_unknown",
                    "condition":f"dual_UU:{h_unknown}:s{hs}+{l_unknown}:s{ls}",
                    "hsi_name":h_unknown,"hsi_severity":hs,
                    "lidar_name":l_unknown,"lidar_severity":ls,
                    "cross_name":None,"cross_severity":0
                })
    for s in ev["dual_severity_levels"]:
        if not quick and bool(ev.get("include_all_known_known_pairs", True)):
            for h in h_known:
                for l in l_known:
                    rows.append({"condition_group":"dual_known_known","condition":f"dual_KK:{h}+{l}:s{s}","hsi_name":h,"hsi_severity":int(s),"lidar_name":l,"lidar_severity":int(s),"cross_name":None,"cross_severity":0})
        if not quick and bool(ev.get("include_unknown_known", True)):
            for l in l_known: rows.append({"condition_group":"dual_unknown_known","condition":f"dual_UK:{h_unknown}+{l}:s{s}","hsi_name":h_unknown,"hsi_severity":int(s),"lidar_name":l,"lidar_severity":int(s),"cross_name":None,"cross_severity":0})
        if not quick and bool(ev.get("include_known_unknown", True)):
            for h in h_known: rows.append({"condition_group":"dual_known_unknown","condition":f"dual_KU:{h}+{l_unknown}:s{s}","hsi_name":h,"hsi_severity":int(s),"lidar_name":l_unknown,"lidar_severity":int(s),"cross_name":None,"cross_severity":0})
        if not uu_matrix:
            rows.append({"condition_group":"dual_unknown_unknown","condition":f"dual_UU:{h_unknown}+{l_unknown}:s{s}","hsi_name":h_unknown,"hsi_severity":int(s),"lidar_name":l_unknown,"lidar_severity":int(s),"cross_name":None,"cross_severity":0})
    if not quick and bool(ev.get("include_misregistration", False)):
        for s in ev.get("misregistration_severity_levels", [1,2,3]):
            rows.append({"condition_group":"misregistration","condition":f"cross:misregistration:s{s}","hsi_name":None,"hsi_severity":0,"lidar_name":None,"lidar_severity":0,"cross_name":"misregistration","cross_severity":int(s)})
    return rows


def mean_group(rows: Sequence[Mapping[str, object]], group: str) -> float:
    vals = [float(r["OA"]) for r in rows if r["condition_group"] == group]
    return float(np.mean(vals)) if vals else float("nan")


def summary(rows: Sequence[Mapping[str, object]], clean_fusion_reference: float, gates: Mapping[str, object]) -> Dict[str, object]:
    clean = mean_group(rows, "clean"); uu = mean_group(rows, "dual_unknown_unknown")
    robust_vals = [float(r["OA"]) for r in rows if r["condition_group"] != "clean"]
    group_means = {
        "hsi_known": mean_group(rows, "hsi_known"),
        "hsi_unknown": mean_group(rows, "hsi_unknown"),
        "lidar_known": mean_group(rows, "lidar_known"),
        "lidar_unknown": mean_group(rows, "lidar_unknown"),
        "dual_known_known": mean_group(rows, "dual_known_known"),
        "dual_unknown_known": mean_group(rows, "dual_unknown_known"),
        "dual_known_unknown": mean_group(rows, "dual_known_unknown"),
        "dual_unknown_unknown": uu,
        "misregistration": mean_group(rows, "misregistration"),
    }
    def macro(names):
        vals = [group_means[n] for n in names if n in group_means and not math.isnan(group_means[n])]
        return float(np.mean(vals)) if vals else float("nan")
    nonclean = [r for r in rows if r["condition_group"] != "clean"]
    worst = min(nonclean, key=lambda r: float(r["OA"])) if nonclean else None
    out = {
        "clean_fusion_reference_OA": clean_fusion_reference,
        "pcrfusion_clean_OA": clean,
        "clean_delta_vs_clean_fusion_pp": (clean - clean_fusion_reference) * 100.0,
        "hsi_known_mOA": group_means["hsi_known"],
        "hsi_unknown_mOA": group_means["hsi_unknown"],
        "lidar_known_mOA": group_means["lidar_known"],
        "lidar_unknown_mOA": group_means["lidar_unknown"],
        "dual_unknown_unknown_mOA": uu,
        "dual_known_known_mOA": group_means["dual_known_known"],
        "dual_unknown_known_mOA": group_means["dual_unknown_known"],
        "dual_known_unknown_mOA": group_means["dual_known_unknown"],
        "misregistration_mOA": group_means["misregistration"],
        "robust_mOA_condition_weighted": float(np.mean(robust_vals)) if robust_vals else float("nan"),
        "single_macro_mOA": macro(["hsi_known", "hsi_unknown", "lidar_known", "lidar_unknown"]),
        "dual_macro_mOA": macro(["dual_known_known", "dual_unknown_known", "dual_known_unknown", "dual_unknown_unknown"]),
        "robust_macro_group_mOA": macro(["hsi_known", "hsi_unknown", "lidar_known", "lidar_unknown", "dual_known_known", "dual_unknown_known", "dual_known_unknown", "dual_unknown_unknown", "misregistration"]),
        "worst_nonclean_condition": str(worst["condition"]) if worst is not None else None,
        "worst_nonclean_OA": float(worst["OA"]) if worst is not None else float("nan"),
    }
    out["clean_gate_pass"] = bool(clean >= float(gates["clean_min_oa"]))
    out["dual_uu_strong_pass"] = bool(uu >= float(gates["dual_uu_target_oa"])) if not math.isnan(uu) else False
    out["dual_uu_stop"] = bool(uu < float(gates["dual_uu_stop_floor"])) if not math.isnan(uu) else False
    return out


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows: return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_uu_matrix(run_dir: Path, rows: Sequence[Mapping[str, object]], h_unknown: str, l_unknown: str) -> None:
    uu = [r for r in rows if r.get("condition_group") == "dual_unknown_unknown"]
    if not uu:
        return
    hs = sorted({int(r["hsi_severity"]) for r in uu})
    ls = sorted({int(r["lidar_severity"]) for r in uu})
    lookup = {(int(r["hsi_severity"]), int(r["lidar_severity"])): float(r["OA"]) for r in uu}
    path = run_dir / "dual_uu_matrix.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([f"{h_unknown}\\{l_unknown}"] + [f"s{l}" for l in ls])
        for h in hs:
            w.writerow([f"s{h}"] + [f"{lookup.get((h,l), float('nan')):.6f}" for l in ls])
    print("\nDual-UU severity matrix (OA %):")
    print("HSI\\LiDAR " + " ".join(f"s{l:>6}" for l in ls))
    for h in hs:
        vals = " ".join(f"{lookup.get((h,l), float('nan'))*100:7.3f}" for l in ls)
        print(f"s{h:<8} {vals}")
    if hs and ls and (max(hs), max(ls)) in lookup:
        print(f"Worst-corner s{max(hs)}+s{max(ls)} OA: {lookup[(max(hs), max(ls))]*100:.3f}%")
    print(f"Matrix saved          : {path}")


def _mean_metric(rows: Sequence[Mapping[str, object]], group: str, metric: str) -> float:
    vals = [float(r[metric]) for r in rows if r.get("condition_group") == group]
    return float(np.mean(vals)) if vals else float("nan")


def build_group_summary(rows: Sequence[Mapping[str, object]], clean_oa: float) -> List[Dict[str, object]]:
    order = [
        "clean", "hsi_known", "hsi_unknown", "lidar_known", "lidar_unknown",
        "dual_known_known", "dual_unknown_known", "dual_known_unknown",
        "dual_unknown_unknown", "misregistration",
    ]
    out: List[Dict[str, object]] = []
    for group in order:
        rr = [r for r in rows if r.get("condition_group") == group]
        if not rr:
            continue
        oas = np.asarray([float(r["OA"]) for r in rr], dtype=float)
        out.append({
            "condition_group": group,
            "count": len(rr),
            "mOA": float(oas.mean()),
            "mAA": float(np.mean([float(r["AA"]) for r in rr])),
            "mKappa": float(np.mean([float(r["Kappa"]) for r in rr])),
            "min_OA": float(oas.min()),
            "max_OA": float(oas.max()),
            "drop_vs_clean_pp": float((clean_oa - oas.mean()) * 100.0),
            "retention_vs_clean_pct": float((oas.mean() / max(clean_oa, 1e-12)) * 100.0),
            "worst_condition": str(rr[int(oas.argmin())]["condition"]),
            "best_condition": str(rr[int(oas.argmax())]["condition"]),
        })
    return out


def render_full_report(rows: Sequence[Mapping[str, object]], group_rows: Sequence[Mapping[str, object]], clean_fusion_reference: float, summ: Mapping[str, object]) -> str:
    clean_rows = [r for r in rows if r.get("condition_group") == "clean"]
    clean = clean_rows[0] if clean_rows else None
    lines: List[str] = []
    lines.append("=" * 104)
    lines.append("PCRFusion FULL ROBUSTNESS REPORT")
    lines.append("=" * 104)
    lines.append(f"Stage-I reference OA : {clean_fusion_reference*100:.3f}%")
    if clean is not None:
        lines.append(
            f"DARFusion clean: OA={float(clean['OA'])*100:.3f}%  AA={float(clean['AA'])*100:.3f}%  "
            f"Kappa={float(clean['Kappa']):.5f}  delta_vs_reference={float(summ['clean_delta_vs_clean_fusion_pp']):+.3f} pp"
        )
        lines.append("PCRFusion clean per-class recall (%): " + ", ".join(f"{100*float(x):.2f}" for x in clean["EachAcc"]))
    lines.append("")
    lines.append("GROUP SUMMARY")
    lines.append(f"{'Group':<24}{'N':>5}{'mOA%':>10}{'mAA%':>10}{'mKappa':>10}{'MinOA%':>10}{'MaxOA%':>10}{'Drop(pp)':>11}{'Retain%':>10}")
    for r in group_rows:
        lines.append(
            f"{str(r['condition_group']):<24}{int(r['count']):>5}"
            f"{float(r['mOA'])*100:>10.3f}{float(r['mAA'])*100:>10.3f}{float(r['mKappa']):>10.5f}"
            f"{float(r['min_OA'])*100:>10.3f}{float(r['max_OA'])*100:>10.3f}"
            f"{float(r['drop_vs_clean_pp']):>11.3f}{float(r['retention_vs_clean_pct']):>10.2f}"
        )
        lines.append(f"    worst={r['worst_condition']} | best={r['best_condition']}")
    lines.append("")
    lines.append("ALL CONDITIONS")
    lines.append(
        f"{'Condition':<52}{'OA%':>9}{'AA%':>9}{'Kappa':>9}{'pCleanH':>10}{'pCleanL':>10}{'repairH':>10}{'repairL':>10}"
    )
    for r in rows:
        lines.append(
            f"{str(r['condition']):<52}{float(r['OA'])*100:>9.3f}{float(r['AA'])*100:>9.3f}"
            f"{float(r['Kappa']):>9.5f}{float(r['mean_hsi_clean_probability']):>10.3f}"
            f"{float(r['mean_lidar_clean_probability']):>10.3f}{float(r['mean_hsi_repair_mass']):>10.3f}"
            f"{float(r['mean_lidar_repair_mass']):>10.3f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    bundle = load_bundle(
        args.config,
        args.data_root,
        args.split_file,
        args.seed,
        args.batch_size,
        args.device,
    )
    root_cfg = bundle["config"]
    cfg = bundle["robust_fusion_runtime"]; exp = cfg["experiment"]; runtime = cfg["runtime"]
    training = copy.deepcopy(cfg["training"]); cfg["training"] = training
    model_cfg = cfg["model"]
    seed = validate_seed(int(root_cfg["seeds"]["model"])); corruption_seed = validate_seed(int(root_cfg["seeds"]["corruption"])); seed_everything(seed)
    device = get_device(str(runtime["device"])); amp = bool(runtime["amp"])
    if args.smoke_test:
        training["robust_training"]["epochs"] = 1

    spec = DatasetSpec.from_config_bundle(bundle, args.data_root)
    dataset = load_dataset(bundle, spec.data_root)
    loaders = make_loaders(dataset, spec)
    eval_name = dataset.metadata.get("split", {}).get("evaluation_split", "configured_test")

    ckpt_path = args.clean_fusion_checkpoint.expanduser().resolve()
    if not ckpt_path.exists(): raise FileNotFoundError(ckpt_path)
    checkpoint = safe_load(ckpt_path)
    clean_fusion = load_clean_fusion_from_checkpoint(checkpoint, dataset.num_bands, dataset.lidar_channels, dataset.num_classes).to(device)
    clean_fusion.eval(); freeze_all(clean_fusion)
    data_signature = signature(dataset, spec)
    if checkpoint.get("data_signature") and checkpoint["data_signature"] != data_signature:
        raise ValueError("Stage-I checkpoint preprocessing/split/data differs from this run")
    if bool(root_cfg["validation"].get("enabled", False)) and not checkpoint.get("data_signature"):
        raise ValueError("Validation mode requires Stage I trained by this version, with matching data_signature")
    clean_fusion_args = dict(checkpoint.get("args", {}))
    if int(clean_fusion_args.get("patch_size", 33)) != spec.patch_size:
        raise ValueError(f"Clean Fusion checkpoint patch_size={clean_fusion_args.get('patch_size')} but dataset config patch_size={spec.patch_size}")
    if int(clean_fusion_args.get("dim", model_cfg["feature_dim"])) != int(model_cfg["feature_dim"]):
        raise ValueError("Clean Fusion feature_dim and Stage-II feature_dim differ")

    clean_fusion_reference = float("nan")
    print(f"[protocol] checkpoint selection uses {eval_name}; held-out families remain excluded.")

    protocol = DualModalDegradationProtocol.from_config_bundle(bundle)
    run_dir = args.output_root.expanduser().resolve() if args.output_root else Path(str(bundle["paths"]["output_root"])).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    shared_cache = provider_cache(bundle, dataset, spec)
    provider = DualModalWholeSceneProvider(
        dataset.hsi, dataset.lidar, shared_cache, spec.patch_size, protocol,
        derived_seed(corruption_seed, "robust_provider"),
    )

    h_unknown = str(exp["hsi_outer_unknown"]); l_unknown = str(exp["lidar_outer_unknown"])
    if h_unknown not in protocol.hsi_degradations or l_unknown not in protocol.lidar_degradations:
        raise ValueError("Configured held-out degradation is missing from the evaluation protocol")
    h_known = [x for x in protocol.hsi_degradations if x != h_unknown]
    l_known = [x for x in protocol.lidar_degradations if x != l_unknown]
    model = PCRFusion(
        clean_fusion, len(h_known), len(l_known),
        feature_dim=int(model_cfg["feature_dim"]), degradation_dim=int(model_cfg["degradation_dim"]),
        degradation_dropout=float(model_cfg["degradation_dropout"]), num_tokens=int(model_cfg["num_degradation_tokens"]),
        num_subprototypes=int(model_cfg["num_subprototypes"]), severity_slots=len(protocol.severity_levels),
        kmeans_iterations=int(model_cfg["kmeans_iterations"]), cosine_temperature=float(model_cfg["cosine_temperature"]),
        expert_hidden_dim=int(model_cfg["expert_hidden_dim"]), expert_residual_scale=float(model_cfg["expert_residual_scale"]),
        repair_hidden_dim=int(model_cfg.get("repair_hidden_dim", 64)),
        repair_dropout=float(model_cfg.get("repair_dropout", 0.10)),
        repair_gain_weight=float(model_cfg.get("repair_gain_weight", 2.0)),
        repair_activation_threshold=float(model_cfg.get("repair_activation_threshold", 0.02)),
        repair_gate_temperature=float(model_cfg.get("repair_gate_temperature", 0.03)),
        hsi_degradation_encoder_variant=str(model_cfg.get("hsi_degradation_encoder_variant", "stats_only")),
        lidar_degradation_encoder_variant=str(model_cfg.get("lidar_degradation_encoder_variant", "statistics")),
        shared_expert_per_modality=bool(model_cfg.get("shared_expert_per_modality", False)),
        composition_config=training["robust_training"].get("composition"),
    ).to(device)
    freeze_all(model)

    print("=" * 80)
    print("DARFusion Top1 | STAGE II: FAMILY EXPERTS + LOW MIX")
    print(f"dataset={dataset.dataset_name} split={eval_name} patch={spec.patch_size} batch={spec.batch_size}")
    print(f"model_seed={seed} corruption_seed={corruption_seed}")
    print(f"clean_fusion_checkpoint={ckpt_path}")
    print(f"known HSI={h_known}; held-out HSI={h_unknown}")
    print(f"known LiDAR={l_known}; held-out LiDAR={l_unknown}")
    print(f"HSI degradation encoder={model.hsi_repair_branch.degradation_encoder_variant}; LiDAR degradation encoder={model.lidar_repair_branch.degradation_encoder_variant}")
    print(f"composition={model.composition_config}; GT expert supervision retained")
    print(f"total params={count_parameters(model):,} ({count_parameters(model)/1e6:.3f} M)")
    print(f"Selection split={eval_name}; score uses clean and known conditions only.")
    print(f"Held-out families: HSI={h_unknown}, LiDAR={l_unknown}.")
    print("=" * 80)

    opt_cfg = training["optimizer"]; weight_decay = float(opt_cfg["weight_decay"]); grad_clip = float(opt_cfg["grad_clip_norm"])
    robust_cfg = training["robust_training"]
    reset_loader_random_streams(loaders, derived_seed(seed, "robust_fusion_loader"))
    training_info = train_robust_fusion_stage(
        model, clean_fusion, loaders["train"], loaders["test"], provider,
        hsi_known_names=h_known, lidar_known_names=l_known,
        device=device, cfg=robust_cfg, weight_decay=weight_decay,
        grad_clip=grad_clip, amp=amp, seed=seed, num_classes=dataset.num_classes,
        history_dir=run_dir, resume=args.resume,
        resume_identity=dict(config=root_cfg,data=data_signature,known_names=dict(hsi=h_known,lidar=l_known)),
    )
    model.set_repairability_ready(False)
    checkpoint_payload = {
        "model": model.state_dict(), "clean_fusion_checkpoint": os.path.relpath(ckpt_path, run_dir), "model_seed": seed,
        "corruption_seed": corruption_seed, "component": "PCRFusion", "held_out": {"hsi": h_unknown, "lidar": l_unknown},
        "hsi_degradation_encoder_variant": model.hsi_repair_branch.degradation_encoder_variant,
        
        "active_route": "top1",
        "selection_route": "cosine_experts",
        "training_policy": "hard_gt_plus_composition" if model.composition_config["enabled"] else "partial_unfreeze_hard_routing",
        "best_epoch": training_info.get("best_epoch"),
        "best_selection": training_info.get("best_selection"),
        "composition_config": model.composition_config,
        "model_config": model_cfg, "clean_fusion_args": clean_fusion_args,
        "data_signature": data_signature, "evaluation_split": eval_name,
        "known_names": {"hsi": h_known, "lidar": l_known},
        "selection_protocol": root_cfg["selection_protocol"],
        "teacher_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "teacher_model_seed": clean_fusion_args.get("seed"),
        "code_signature": code_signature(),
    }
    checkpoint_payload["training_policy"] = "hard_gt_plus_composition" if model.composition_config["enabled"] else "partial_unfreeze_hard_routing"
    atomic_torch(run_dir / "best.pt",checkpoint_payload)
    (run_dir / "resolved_config.json").write_text(json.dumps({"experiment": root_cfg, "paths": bundle["paths"], "hashes": bundle["hashes"], "dataset_metadata": dataset.metadata}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.skip_final_evaluation:
        print(f"Saved checkpoint and training history: {run_dir}")
        return

    final_loader, final_split = evaluation_loader(bundle, dataset, spec, loaders["test"], split="official_test")
    clean_fusion_reference_metrics = evaluate_clean_fusion(clean_fusion, final_loader, device, dataset.num_classes)
    clean_fusion_reference = float(clean_fusion_reference_metrics["OA"])
    print(f"[reference] Stage-I clean OA={clean_fusion_reference*100:.3f}%")

    quick = args.eval_profile == "core"
    defs = condition_rows(protocol, h_unknown, l_unknown, cfg["evaluation"], quick, True)

    journal=EvaluationJournal(run_dir/'evaluation_progress.json',dict(model=state_hash(model.state_dict()),data=data_signature,route='top1',profile=args.eval_profile,definitions=defs),defs,args.resume)
    rows=list(journal.rows)
    if rows: print(f"[resume:evaluation] {len(rows)}/{len(defs)} conditions already complete",flush=True)
    for c in defs[len(rows):]:
        m = evaluate_condition(model, final_loader, provider, device=device, num_classes=dataset.num_classes,
                               hsi_name=c["hsi_name"], hsi_severity=int(c["hsi_severity"]),
                               lidar_name=c["lidar_name"], lidar_severity=int(c["lidar_severity"]),
                               cross_name=c["cross_name"], cross_severity=int(c["cross_severity"]))
        row = {**c, **m}; journal.add(row); rows.append(row)
        print(f"[evaluate:{final_split}] {c['condition']:<45} OA={float(m['OA']):.5f} AA={float(m['AA']):.5f} K={float(m['Kappa']):.5f}")

    write_csv(run_dir / "conditions.csv", rows)
    clean_oa = mean_group(rows, "clean")
    group_rows = build_group_summary(rows, clean_oa)
    write_csv(run_dir / "groups.csv", group_rows)
    write_uu_matrix(run_dir, rows, h_unknown, l_unknown)
    summ = summary(rows, clean_fusion_reference, cfg.get("gates", {}))
    nonclean = [float(row["OA"]) for row in rows if row["condition_group"] != "clean"]
    summ.update({
        "nonclean_macro_OA": float(np.mean(nonclean)),
        "nonclean_worst_OA": float(min(nonclean)),
        "evaluation_split": final_split,
        "selection_protocol": root_cfg["selection_protocol"],
        
        "training_policy": "hard_gt_plus_composition" if model.composition_config["enabled"] else "partial_unfreeze_hard_routing",
        "active_route": "top1",
        "selection_route": "cosine_experts",
        "eval_profile": args.eval_profile,
        "robust_training_best_epoch": training_info.get("best_epoch"),
        "robust_training_best_selection": training_info.get("best_selection"),
    })
    report_text = render_full_report(rows, group_rows, clean_fusion_reference, summ)
    (run_dir / "report.txt").write_text(report_text, encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + report_text)
    print("="*80)
    print("PCRFusion FINAL RESULT")
    print(f"Stage-I reference OA  : {clean_fusion_reference*100:.3f}%")
    print(f"PCRFusion clean       : {float(summ['pcrfusion_clean_OA'])*100:.3f}%")
    print(f"HSI unknown mOA      : {float(summ['hsi_unknown_mOA'])*100:.3f}%")
    print(f"LiDAR unknown mOA    : {float(summ['lidar_unknown_mOA'])*100:.3f}%")
    print(f"Dual UU mOA          : {float(summ['dual_unknown_unknown_mOA'])*100:.3f}%")
    print(f"Saved                : {run_dir}")
    print("="*80)

    resolved = {
        "experiment": root_cfg,
        "training_policy": "hard_gt_plus_composition" if model.composition_config["enabled"] else "partial_unfreeze_hard_routing",
        "paths": bundle["paths"],
        "hashes": bundle["hashes"],
        "dataset_metadata": dataset.metadata,
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
