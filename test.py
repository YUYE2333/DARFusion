from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from datasets.registry import load_dataset
from datasets.common import make_loaders, seed_everything
from datasets.spec import DatasetSpec
from degradation.protocol import DualModalDegradationProtocol
from degradation.provider import DualModalWholeSceneProvider
from engine.train_robust_fusion import (
    build_group_summary,
    condition_rows,
    evaluate_condition,
    freeze_all,
    mean_group,
    render_full_report,
    summary,
    write_csv,
    write_uu_matrix,
)
from models.clean_fusion import CleanFusionBackbone
from models.pcrfusion import PCRFusion
from random_control import derived_seed
from train import preflight
from utils.config import load_experiment
from utils.data_signature import signature, provider_cache, evaluation_loader
from utils.run_output import create_unique_run_dir


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "experiments" / "houston2013.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one PCRFusion checkpoint with fixed Top1 inference")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-profile", choices=("core", "full"), default="full")
    return parser.parse_args()


def _write_summary_csv(path: Path, row: dict) -> None:
    values = {
        key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
        for key, value in row.items()
    }
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(values)


def main():
    args = parse_args()
    resolved = load_experiment(
        args.config,
        data_root_override=args.data_root,
        split_file_override=args.split_file,
        output_root_override=args.output_root,
        device_override=args.device,
    )
    preflight(resolved)
    cfg = resolved["config"]
    seed = int(cfg["seeds"]["model"])
    corruption_seed = int(cfg["seeds"]["corruption"])
    seed_everything(seed)
    device_name = str(cfg["runtime"].get("device", "cuda"))
    device = torch.device(device_name if not device_name.startswith("cuda") or torch.cuda.is_available() else "cpu")
    spec = DatasetSpec.from_config_bundle(resolved, args.data_root)
    dataset = load_dataset(resolved, spec.data_root)
    loaders = make_loaders(dataset, spec)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("The supplied file is not a PCRFusion checkpoint containing a model state.")

    if checkpoint.get('data_signature') and checkpoint['data_signature'] != signature(dataset, spec):
        raise ValueError('Checkpoint data/preprocessing/split signature differs. Reuse the original data and split paths.')
    if checkpoint.get('held_out') and checkpoint['held_out'] != {'hsi':resolved['experiment']['hsi_outer_unknown'], 'lidar':resolved['experiment']['lidar_outer_unknown']}:
        raise ValueError('Checkpoint held-out families differ from the supplied configuration.')
    final_loader, final_split = evaluation_loader(resolved, dataset, spec, loaders['test'], split='official_test')
    variant = str(checkpoint.get('hsi_degradation_encoder_variant') or cfg['robust_fusion']['model'].get('hsi_degradation_encoder_variant','stats_only'))
    if variant != 'stats_only': raise ValueError('The selected model uses HSI stats_only.')
    clean_fusion_cfg = cfg["clean_fusion"]
    pca_cfg = dict(clean_fusion_cfg.get("pca", {}))
    backbone_args = dict(checkpoint.get("clean_fusion_args", {}))
    if "patch_size" in backbone_args and int(backbone_args["patch_size"]) != spec.patch_size:
        raise ValueError("Checkpoint patch size differs from the supplied configuration.")
    clean_fusion = CleanFusionBackbone(
        dataset.num_bands,
        dataset.lidar_channels,
        dataset.num_classes,
        dim=int(backbone_args.get("dim", clean_fusion_cfg.get("feature_dim", 96))),
        heads=int(backbone_args.get("heads", clean_fusion_cfg.get("heads", 4))),
        dropout=float(backbone_args.get("dropout", clean_fusion_cfg.get("dropout", 0.1))),
        fusion=str(backbone_args.get("fusion", clean_fusion_cfg.get("fusion", "cross_attention"))),
        mixstyle_p=float(backbone_args.get("mixstyle_p", clean_fusion_cfg.get("mixstyle_probability", 0.0))),
        pca_components=int(backbone_args.get("pca_components", int(pca_cfg.get("components", 20)) if bool(pca_cfg.get("enabled", True)) else 0)),
    ).to(device)

    protocol = DualModalDegradationProtocol.from_config_bundle(resolved)
    exp = resolved["experiment"]
    h_unknown = str(exp["hsi_outer_unknown"])
    l_unknown = str(exp["lidar_outer_unknown"])
    h_known = [name for name in protocol.hsi_degradations if name != h_unknown]
    l_known = [name for name in protocol.lidar_degradations if name != l_unknown]
    model_cfg = checkpoint.get("model_config") or cfg["robust_fusion"]["model"]
    if model_cfg.get("shared_expert_per_modality", False): raise ValueError("A shared-expert checkpoint is not the selected model.")
    if checkpoint.get("known_names") and checkpoint["known_names"] != {"hsi":h_known, "lidar":l_known}: raise ValueError("Checkpoint expert-family order differs from the configuration.")
    model = PCRFusion(
        clean_fusion,
        len(h_known),
        len(l_known),
        feature_dim=int(model_cfg["feature_dim"]),
        degradation_dim=int(model_cfg["degradation_dim"]),
        degradation_dropout=float(model_cfg["degradation_dropout"]),
        num_tokens=int(model_cfg["num_degradation_tokens"]),
        num_subprototypes=int(model_cfg["num_subprototypes"]),
        severity_slots=len(protocol.severity_levels),
        kmeans_iterations=int(model_cfg["kmeans_iterations"]),
        cosine_temperature=float(model_cfg["cosine_temperature"]),
        expert_hidden_dim=int(model_cfg["expert_hidden_dim"]),
        expert_residual_scale=float(model_cfg["expert_residual_scale"]),
        repair_hidden_dim=int(model_cfg.get("repair_hidden_dim", 64)),
        repair_dropout=float(model_cfg.get("repair_dropout", 0.1)),
        repair_gain_weight=float(model_cfg.get("repair_gain_weight", 2.0)),
        repair_activation_threshold=float(model_cfg.get("repair_activation_threshold", 0.02)),
        repair_gate_temperature=float(model_cfg.get("repair_gate_temperature", 0.03)),
        hsi_degradation_encoder_variant=variant,
        lidar_degradation_encoder_variant=str(model_cfg.get("lidar_degradation_encoder_variant", "statistics")),
        composition_config=checkpoint.get('composition_config'),
        shared_expert_per_modality=bool(model_cfg.get('shared_expert_per_modality', False)),
    ).to(device)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except Exception as exc:
        raise RuntimeError(f"Checkpoint is not compatible with PCRFusion: {checkpoint_path}") from exc
    model.set_repairability_ready(False)
    model.eval()
    freeze_all(model)

    output_root = Path(str(resolved["paths"]["output_root"]))
    run_dir = create_unique_run_dir(output_root, dataset.dataset_name, seed, run_name=args.run_name, section="DARFusion_Top1_test")
    shared_cache = provider_cache(resolved, dataset, spec)
    provider = DualModalWholeSceneProvider(
        dataset.hsi,
        dataset.lidar,
        shared_cache,
        spec.patch_size,
        protocol,
        derived_seed(corruption_seed, "robust_provider"),
    )

    definitions = condition_rows(
        protocol,
        h_unknown,
        l_unknown,
        cfg["evaluation"],
        args.eval_profile == "core",
        True,
    )
    rows = []
    for condition in definitions:
        metrics = evaluate_condition(
            model,
            final_loader,
            provider,
            device=device,
            num_classes=dataset.num_classes,
            hsi_name=condition["hsi_name"],
            hsi_severity=int(condition["hsi_severity"]),
            lidar_name=condition["lidar_name"],
            lidar_severity=int(condition["lidar_severity"]),
            cross_name=condition["cross_name"],
            cross_severity=int(condition["cross_severity"]),
        )
        rows.append({**condition, **metrics})
        print(f"[test] {condition['condition']:<45} OA={float(metrics['OA']):.5f}")

    clean_oa = mean_group(rows, "clean")
    group_rows = build_group_summary(rows, clean_oa)
    result = summary(rows, float("nan"), cfg.get("gates", {}))
    result.update({
        "method": "DARFusion",
        "dataset": dataset.dataset_name,
        "model_seed": seed,
        "corruption_seed": corruption_seed,
        "checkpoint": str(checkpoint_path),
        "training_policy": str(checkpoint.get("training_policy", "unknown")),
        "active_route": "top1",
        "evaluation_split": final_split,
        "selection_protocol": checkpoint.get("selection_protocol", cfg.get("selection_protocol", "official_test_guided")),
        "nonclean_macro_OA": sum(float(row["OA"]) for row in rows if row["condition_group"] != "clean") / sum(row["condition_group"] != "clean" for row in rows),
        "nonclean_worst_OA": min(float(row["OA"]) for row in rows if row["condition_group"] != "clean"),
        "eval_profile": args.eval_profile,
        "run_directory": str(run_dir),
    })
    write_csv(run_dir / "conditions.csv", rows)
    write_csv(run_dir / "groups.csv", group_rows)
    write_uu_matrix(run_dir, rows, h_unknown, l_unknown)
    report = render_full_report(rows, group_rows, float("nan"), result).replace("PCRFusion", "DARFusion")
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(run_dir / "summary.csv", result)

    print("\n" + report)
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
