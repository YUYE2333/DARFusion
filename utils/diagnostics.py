from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


def write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    from utils.resume import atomic_rows
    atomic_rows(path,rows)


def write_matrix(path: Path, matrix: Sequence[Sequence[Any]], labels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *labels])
        for label, row in zip(labels, arr.tolist()):
            w.writerow([label, *row])


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def classification_detail(labels: np.ndarray, predictions: np.ndarray, class_names: Sequence[str]) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for y, p in zip(labels, predictions):
        if 0 <= int(y) < n and 0 <= int(p) < n:
            cm[int(y), int(p)] += 1
    per_class = []
    recalls = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        predicted = int(cm[:, i].sum())
        precision = _safe_div(tp, predicted)
        recall = _safe_div(tp, support)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        recalls.append(recall)
        per_class.append({
            "class_id": i,
            "class_name": str(name),
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return {
        "OA": float(np.mean(labels == predictions)) if labels.size else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def deterministic_coordinate_subset(dataset, max_samples: int = 128) -> Tensor:
    coords = np.asarray(dataset.coordinates, dtype=np.int64)
    if coords.shape[0] == 0:
        return torch.empty((0, 2), dtype=torch.long)
    n = min(int(max_samples), int(coords.shape[0]))
    if n == coords.shape[0]:
        chosen = coords
    else:
        idx = np.linspace(0, coords.shape[0] - 1, n, dtype=np.int64)
        chosen = coords[idx]
    return torch.from_numpy(np.asarray(chosen, dtype=np.int64))


@torch.no_grad()
def degradation_encoder_diagnostic(
    branch,
    provider,
    *,
    modality: str,
    known_names: Sequence[str],
    coordinates: Tensor,
    device: torch.device,
    max_chunk: int = 32,
) -> Dict[str, Any]:
    """Fixed, balanced diagnostic. It never changes model parameters or RNG streams.

    Each type (clean + every known family) receives the same number of severity slots.
    Clean is intentionally repeated once per severity slot so type support is balanced.
    """
    old_training = bool(branch.degradation_encoder.training)
    branch.degradation_encoder.eval()
    severities = [int(x) for x in provider.protocol.severity_levels]
    max_sev = max(severities) if severities else 1
    conditions: List[tuple[str | None, int, int, str]] = []
    for sev in severities:
        conditions.append((None, 0, int(branch.clean_type_id), "clean"))
    for tid, name in enumerate(known_names):
        for sev in severities:
            conditions.append((str(name), int(sev), int(tid), str(name)))

    true_types: List[np.ndarray] = []
    pred_types: List[np.ndarray] = []
    true_sev: List[np.ndarray] = []
    pred_sev: List[np.ndarray] = []
    family_labels: List[np.ndarray] = []
    pclean_rows: List[Dict[str, Any]] = []

    coords = coordinates.cpu()
    for name, sev, tid, family_name in conditions:
        condition_preds = []
        condition_pclean = []
        condition_severity = []
        for start in range(0, int(coords.shape[0]), max(1, int(max_chunk))):
            chunk = coords[start:start + max_chunk]
            rows = chunk[:, 0]
            cols = chunk[:, 1]
            hn = name if modality == "hsi" else None
            ln = name if modality == "lidar" else None
            h, l = provider.extract_condition(rows, cols, device, hn, sev if hn else 0, ln, sev if ln else 0)
            x = h if modality == "hsi" else l
            out = branch.degradation_encoder(x)
            condition_preds.append(out["type_logits"].argmax(1).cpu().numpy())
            condition_pclean.append(out["clean_probability"].cpu().numpy())
            condition_severity.append(out["severity"].cpu().numpy())
        if not condition_preds:
            continue
        pp = np.concatenate(condition_preds)
        pc = np.concatenate(condition_pclean)
        ps = np.concatenate(condition_severity)
        count = pp.shape[0]
        target_sev = 0.0 if name is None else float(sev) / float(max_sev)
        true_types.append(np.full(count, tid, dtype=np.int64))
        pred_types.append(pp.astype(np.int64, copy=False))
        true_sev.append(np.full(count, target_sev, dtype=np.float32))
        pred_sev.append(ps.astype(np.float32, copy=False))
        family_labels.append(np.full(count, family_name, dtype=object))
        pclean_rows.append({
            "condition": "clean" if name is None else f"{name}:s{sev}",
            "family": family_name,
            "severity": 0 if name is None else int(sev),
            "type_id": tid,
            "support": count,
            "type_accuracy": float(np.mean(pp == tid)),
            "mean_pclean": float(np.mean(pc)),
            "severity_mae": float(np.mean(np.abs(ps - target_sev))),
            "mean_predicted_severity": float(np.mean(ps)),
            "target_severity": target_sev,
        })

    y = np.concatenate(true_types) if true_types else np.empty(0, dtype=np.int64)
    p = np.concatenate(pred_types) if pred_types else np.empty(0, dtype=np.int64)
    ys = np.concatenate(true_sev) if true_sev else np.empty(0, dtype=np.float32)
    ps = np.concatenate(pred_sev) if pred_sev else np.empty(0, dtype=np.float32)
    fam = np.concatenate(family_labels) if family_labels else np.empty(0, dtype=object)
    names = [str(x) for x in known_names] + ["clean"]
    detail = classification_detail(y, p, names)

    severity_rows: List[Dict[str, Any]] = []
    for family in [str(x) for x in known_names]:
        mask = fam == family
        if np.any(mask):
            severity_rows.append({
                "family": family,
                "support": int(mask.sum()),
                "severity_mae": float(np.mean(np.abs(ps[mask] - ys[mask]))),
                "mean_predicted_severity": float(np.mean(ps[mask])),
                "mean_target_severity": float(np.mean(ys[mask])),
            })
    degraded = fam != "clean"
    severity_mae = float(np.mean(np.abs(ps[degraded] - ys[degraded]))) if np.any(degraded) else float("nan")
    if old_training:
        branch.degradation_encoder.train()
    return {
        "type_OA": detail["OA"],
        "balanced_type_accuracy": detail["balanced_accuracy"],
        "severity_mae": severity_mae,
        "confusion_matrix": detail["confusion_matrix"],
        "per_class": detail["per_class"],
        "condition_rows": pclean_rows,
        "severity_rows": severity_rows,
        "class_names": names,
        "diagnostic_samples_per_condition": int(coords.shape[0]),
    }


def save_degradation_diagnostic(root: Path, modality: str, epoch: int, train_row: Mapping[str, Any], diag: Mapping[str, Any], final: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    history_path = root / f"{modality}_epoch_metrics.csv"
    existing: List[Dict[str, Any]] = []
    if history_path.exists():
        with history_path.open("r", newline="", encoding="utf-8-sig") as f:
            existing = list(csv.DictReader(f))
    row = {
        "epoch": int(epoch),
        **dict(train_row),
        "fixed_type_OA": float(diag["type_OA"]),
        "fixed_balanced_type_accuracy": float(diag["balanced_type_accuracy"]),
        "fixed_severity_mae": float(diag["severity_mae"]),
        "diagnostic_samples_per_condition": int(diag["diagnostic_samples_per_condition"]),
    }
    existing = [r for r in existing if str(r.get("epoch")) != str(epoch)]
    existing.append(row)
    existing.sort(key=lambda r: int(float(r["epoch"])))
    write_rows(history_path, existing)
    if final:
        write_matrix(root / f"{modality}_confusion_matrix.csv", diag["confusion_matrix"], diag["class_names"])
        write_rows(root / f"{modality}_per_class_metrics.csv", diag["per_class"])
        write_rows(root / f"{modality}_severity_metrics.csv", diag["severity_rows"])
        write_rows(root / f"{modality}_condition_metrics.csv", diag["condition_rows"])
        (root / f"{modality}_final_metrics.json").write_text(json.dumps({
            "type_OA": diag["type_OA"],
            "balanced_type_accuracy": diag["balanced_type_accuracy"],
            "severity_mae": diag["severity_mae"],
            "diagnostic_samples_per_condition": diag["diagnostic_samples_per_condition"],
        }, indent=2), encoding="utf-8")
