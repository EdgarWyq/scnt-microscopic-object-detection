from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


CLASS_NAMES = {
    0: "injection_needle",
    1: "holding_needle",
    2: "oocyte",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str, root: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    base = root if root is not None else project_root()
    return (base / p).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def yolo_label_path_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    rel = image_path.relative_to(images_dir)
    return labels_dir / rel.with_suffix(".txt")


def safe_rmtree(path: Path) -> None:
    root = project_root().resolve()
    target = path.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("Refusing to remove a path outside the project: {}".format(target))
    if target.exists():
        shutil.rmtree(str(target))


def link_or_copy_file(src: Path, dst: Path, mode: str = "copy") -> str:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(str(src), str(dst))
        return "copied"

    if mode == "hardlink":
        try:
            os.link(str(src), str(dst))
            return "hardlinked"
        except OSError:
            shutil.copy2(str(src), str(dst))
            return "copied"

    if mode in {"symlink", "auto"}:
        try:
            os.symlink(str(src), str(dst))
            return "symlinked"
        except OSError:
            if mode == "symlink":
                raise
            shutil.copy2(str(src), str(dst))
            return "copied"

    raise ValueError("Unsupported link mode: {}".format(mode))


def copy_tree_flat_by_relative(
    src_files: Sequence[Path],
    src_root: Path,
    dst_root: Path,
    mode: str = "copy",
) -> Dict[str, int]:
    counts = {"copied": 0, "symlinked": 0, "hardlinked": 0}
    for src in src_files:
        dst = dst_root / src.relative_to(src_root)
        action = link_or_copy_file(src, dst, mode)
        counts[action] = counts.get(action, 0) + 1
    return counts


def _relative_or_absolute(path: Path, base: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_dataset_yaml(yaml_path: Path, train_images: Path, val_images: Path, root: Optional[Path] = None) -> None:
    root = root or project_root()
    ensure_dir(yaml_path.parent)
    text = "\n".join(
        [
            "# Auto-generated SCNT dataset config for Ultralytics YOLO.",
            "path: {}".format(root.resolve().as_posix()),
            "train: {}".format(_relative_or_absolute(train_images, root)),
            "val: {}".format(_relative_or_absolute(val_images, root)),
            "test:",
            "nc: 3",
            "names:",
            "  0: injection_needle",
            "  1: holding_needle",
            "  2: oocyte",
            "",
        ]
    )
    yaml_path.write_text(text, encoding="utf-8")


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value) -> List:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            out = value.tolist()
            return out if isinstance(out, list) else [out]
        return list(value)
    except Exception:
        return []


def extract_yolo_metrics(metrics, experiment: str, model_path: Path, data_yaml: Path) -> Dict[str, object]:
    box = getattr(metrics, "box", metrics)
    row: Dict[str, object] = {
        "timestamp": now_timestamp(),
        "experiment": experiment,
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "map50": _as_float(getattr(box, "map50", None)),
        "map50_95": _as_float(getattr(box, "map", None)),
    }

    ap50_all = _as_list(getattr(box, "ap50", None))
    maps_all = _as_list(getattr(box, "maps", None))

    for class_id, class_name in CLASS_NAMES.items():
        ap50 = None
        ap = None
        if hasattr(box, "class_result"):
            try:
                class_result = box.class_result(class_id)
                if len(class_result) >= 4:
                    ap50 = _as_float(class_result[2])
                    ap = _as_float(class_result[3])
            except Exception:
                pass
        if ap50 is None and class_id < len(ap50_all):
            ap50 = _as_float(ap50_all[class_id])
        if ap is None and class_id < len(maps_all):
            ap = _as_float(maps_all[class_id])
        row["ap50_{}".format(class_name)] = ap50
        row["map50_95_{}".format(class_name)] = ap

    return row


def append_csv_row(csv_path: Path, row: Dict[str, object]) -> None:
    ensure_dir(csv_path.parent)
    existing: List[Dict[str, object]] = []
    fieldnames: List[str] = []
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing = list(reader)

    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for old in existing:
            writer.writerow({k: old.get(k, "") for k in fieldnames})
        writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fieldnames})


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest_rows_by_experiment(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    latest: Dict[str, Dict[str, str]] = {}
    for row in rows:
        exp = row.get("experiment", "")
        if not exp:
            continue
        latest[exp] = row
    return latest


def update_marked_section(markdown_path: Path, start_marker: str, end_marker: str, replacement: str) -> None:
    if markdown_path.exists():
        text = markdown_path.read_text(encoding="utf-8")
    else:
        text = ""
    if start_marker in text and end_marker in text:
        before, rest = text.split(start_marker, 1)
        _, after = rest.split(end_marker, 1)
        text = before + start_marker + "\n" + replacement.rstrip() + "\n" + end_marker + after
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n{}\n{}\n{}\n".format(start_marker, replacement.rstrip(), end_marker)
    markdown_path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

