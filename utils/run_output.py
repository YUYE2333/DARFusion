from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional


def _safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip())
    return value.strip("._-") or "run"


def create_unique_run_dir(
    output_root: Path,
    dataset: str,
    seed: int,
    *,
    run_name: Optional[str] = None,
    section: str = "DARFusion",
) -> Path:
    """Create an atomic, never-overwritten run directory.

    Reusing the same seed is intentional and does not imply resuming or replacing
    an earlier run.  A timestamp is used by default; an incrementing suffix also
    protects two launches in the same second and repeated explicit run names.
    """
    parent = Path(output_root).expanduser().resolve() / _safe_name(dataset) / _safe_name(section)
    parent.mkdir(parents=True, exist_ok=True)
    base = _safe_name(run_name) if run_name else f"{datetime.now():%Y%m%d_%H%M%S}_seed{int(seed)}"
    for index in range(1, 10000):
        name = base if index == 1 else f"{base}_{index:02d}"
        candidate = parent / name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a unique run directory below {parent}")


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else {}


def _write_one_row_csv(path: Path, row: Mapping[str, Any]) -> None:
    values = {
        str(key): (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
        for key, value in row.items()
    }
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(values)


def finalize_training_run(
    run_dir: Path,
    *,
    clean_fusion_checkpoint: Path,
    clean_fusion_raw: Optional[Path],
    pcrfusion_raw: Path,
    metadata: Mapping[str, Any],
    keep_work: bool = False,
) -> dict:
    """Flatten the useful artifacts and remove successful staging output."""
    run_dir = Path(run_dir).resolve()
    _copy_if_exists(Path(clean_fusion_checkpoint), run_dir / "clean_fusion.pt")
    _copy_if_exists(pcrfusion_raw / "best.pt", run_dir / "pcrfusion.pt")
    _copy_if_exists(pcrfusion_raw / "report.txt", run_dir / "report.txt")
    _copy_if_exists(pcrfusion_raw / "conditions.csv", run_dir / "conditions.csv")
    _copy_if_exists(pcrfusion_raw / "groups.csv", run_dir / "groups.csv")
    _copy_if_exists(pcrfusion_raw / "dual_uu_matrix.csv", run_dir / "uu_matrix.csv")
    _copy_if_exists(pcrfusion_raw / "pcrfusion_history.csv", run_dir / "pcrfusion_history.csv")
    _copy_if_exists(pcrfusion_raw / "resolved_config.json", run_dir / "run_config.json")
    if clean_fusion_raw is not None:
        _copy_if_exists(clean_fusion_raw / "history.csv", run_dir / "clean_fusion_history.csv")
        _copy_if_exists(clean_fusion_raw / "metrics.json", run_dir / "clean_fusion_metrics.json")

    # Preserve legacy filenames for existing tooling; DARFusion is the public name.
    _copy_if_exists(run_dir / "pcrfusion.pt", run_dir / "darfusion.pt")
    _copy_if_exists(run_dir / "pcrfusion_history.csv", run_dir / "darfusion_history.csv")
    report_path = run_dir / "report.txt"
    if report_path.is_file():
        report_path.write_text(report_path.read_text(encoding="utf-8").replace("PCRFusion", "DARFusion"), encoding="utf-8")
    summary = {**dict(metadata), **_read_json(pcrfusion_raw / "summary.json")}
    summary['method'] = 'DARFusion'
    if 'pcrfusion_clean_OA' in summary:
        summary['darfusion_clean_OA'] = summary['pcrfusion_clean_OA']
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _write_one_row_csv(run_dir / "summary.csv", summary)
    (run_dir / "run_info.json").write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    work = run_dir / ".work"
    if work.is_dir() and not keep_work:
        if work.resolve().parent != run_dir:
            raise RuntimeError(f"Refusing to clean staging directory outside run: {work}")
        shutil.rmtree(work)
    return summary


__all__ = ["create_unique_run_dir", "finalize_training_run"]
