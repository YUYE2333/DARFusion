from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from utils.config import load_experiment
from utils.final_config import materialize
from utils.run_output import create_unique_run_dir, finalize_training_run


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "experiments" / "houston2013.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the selected DARFusion Top1 model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--clean-fusion-checkpoint", dest="clean_fusion_checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", default=None, help="Optional readable name; duplicates receive _02, _03, ...")
    parser.add_argument("--eval-profile", choices=("core", "full"), default="full")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--selection", choices=("official_test",), default="official_test")
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    return parser.parse_args()


def run(command):
    print("\n" + "=" * 100)
    print("RUN:", " ".join(str(value) for value in command), flush=True)
    print("=" * 100)
    subprocess.run([str(value) for value in command], check=True, cwd=HERE)


def preflight(resolved):
    root = Path(str(resolved["paths"]["dataset_root"]))
    split = Path(str(resolved["paths"]["label_split_file"]))
    expected = [root / str(name) for name in resolved["dataset"]["files"].values()] + [split]
    missing = [path for path in expected if not path.is_file()]
    if not missing:
        return
    project = Path(str(resolved["paths"]["project_root"]))
    display = []
    for path in missing:
        try:
            display.append(str(path.relative_to(project)))
        except ValueError:
            display.append(str(path))
    raise FileNotFoundError(
        "DARFusion data preflight failed. Missing:\n  - " + "\n  - ".join(display)
        + "\nPut data under <project>/data/<dataset>/ and split MAT files under <project>/splits/."
    )


def _append_overrides(command, args, *, robust_stage: bool = False):
    if args.data_root is not None:
        command += ["--data-root", args.data_root]
    if args.split_file is not None:
        command += ["--split-file", args.split_file]
    if args.model_seed is not None:
        command += ["--seed" if robust_stage else "--model-seed", args.model_seed]
    if args.batch_size is not None:
        command += ["--batch-size", args.batch_size]
    if args.device is not None:
        command += ["--device", args.device]
    if args.smoke_test:
        command += ["--smoke-test"]
    return command


def main():
    args = parse_args()
    source_config = args.config.expanduser().resolve()
    args.config = materialize(source_config, selection=args.selection, data_root=args.data_root,
        split_file=args.split_file, model_seed=args.model_seed, batch_size=args.batch_size,
        device=args.device, stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs)
    resolved = load_experiment(
        args.config,
        data_root_override=args.data_root,
        split_file_override=args.split_file,
        output_root_override=args.output_root,
        model_seed_override=args.model_seed,
        batch_size_override=args.batch_size,
        device_override=args.device,
    )
    preflight(resolved)
    if args.clean_fusion_checkpoint is not None and not args.clean_fusion_checkpoint.expanduser().resolve().is_file():
        raise FileNotFoundError(args.clean_fusion_checkpoint)

    cfg = resolved["config"]
    dataset = str(resolved["dataset"]["dataset"])
    seed = int(cfg["seeds"]["model"])
    output_root = Path(str(resolved["paths"]["output_root"]))
    run_dir = create_unique_run_dir(output_root, dataset, seed, run_name=args.run_name, section="DARFusion_Top1")
    work = run_dir / ".work"
    clean_fusion_raw = work / "clean_fusion"
    pcrfusion_raw = work / "pcrfusion"

    print(f"[DARFusion Top1] run directory: {run_dir}")
    print(f"Stage I: epochs={cfg['clean_fusion']['epochs']} lr={cfg['clean_fusion']['learning_rate']} wd={cfg['clean_fusion']['weight_decay']} patch={resolved['dataset']['patch_size']}")
    print("Stage II: original hard routing + low mix=0.05; selection=Full; final inference=Top1")
    if args.clean_fusion_checkpoint is None:
        command = [
            sys.executable, "-m", "engine.train_clean_fusion",
            "--config", args.config,
            "--output-root", clean_fusion_raw,
        ]
        run(_append_overrides(command, args))
        clean_fusion_checkpoint = clean_fusion_raw / "best.pt"
    else:
        clean_fusion_raw = None
        clean_fusion_checkpoint = args.clean_fusion_checkpoint.expanduser().resolve()

    command = [
        sys.executable, "-m", "engine.train_robust_fusion",
        "--config", args.config,
        "--clean-fusion-checkpoint", clean_fusion_checkpoint,
        "--output-root", pcrfusion_raw,
        "--eval-profile", args.eval_profile,
    ]
    run(_append_overrides(command, args, robust_stage=True))

    metadata = {
        "project": "DARFusion Top1",
        "source_config": str(source_config),
        "effective_config": str(args.config),
        "active_route": "top1",
        "selection_route": "cosine_experts",
        "dataset": dataset,
        "model_seed": seed,
        "split_seed": int(cfg["seeds"]["split"]),
        "corruption_seed": int(cfg["seeds"]["corruption"]),
        "batch_size": int(cfg["runtime"]["batch_size"]),
        "patch_size": int(resolved["dataset"]["patch_size"]),
        "fold_name": str(resolved["experiment"]["fold_name"]),
        "hsi_unknown": str(resolved["experiment"]["hsi_outer_unknown"]),
        "lidar_unknown": str(resolved["experiment"]["lidar_outer_unknown"]),
        "eval_profile": args.eval_profile,
        "selection_protocol": str(cfg.get("selection_protocol", "official_test_guided")),
        "clean_fusion_checkpoint_source": str(run_dir / "clean_fusion.pt"),
        "run_directory": str(run_dir),
    }
    summary = finalize_training_run(
        run_dir,
        clean_fusion_checkpoint=clean_fusion_checkpoint,
        clean_fusion_raw=clean_fusion_raw,
        pcrfusion_raw=pcrfusion_raw,
        metadata=metadata,
    )
    (run_dir / "resolved_experiment.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("DARFusion Top1 FINISHED")
    print(f"Clean Fusion OA : {float(summary['clean_fusion_reference_OA']) * 100:.3f}%")
    print(f"DARFusion OA   : {float(summary['pcrfusion_clean_OA']) * 100:.3f}%")
    print(f"HSI unknown   : {float(summary['hsi_unknown_mOA']) * 100:.3f}%")
    print(f"LiDAR unknown : {float(summary['lidar_unknown_mOA']) * 100:.3f}%")
    print(f"Dual UU mOA   : {float(summary['dual_unknown_unknown_mOA']) * 100:.3f}%")
    print(f"Saved         : {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
