from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from common import CLASS_NAMES, ensure_dir, link_or_copy_file, list_images, project_root, resolve_path, safe_rmtree


COLORS = {
    0: (40, 200, 255),
    1: (255, 120, 40),
    2: (80, 220, 80),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-confidence pseudo labels for target-domain images.")
    parser.add_argument("--model", required=True, help="Baseline YOLO weights, e.g. runs/scnt/baseline/weights/best.pt.")
    parser.add_argument("--image-dir", default="dataset/SCNT/target_split/target_adapt/images", help="Unlabeled target images for adaptation.")
    parser.add_argument("--transductive-all-target", action="store_true", help="Use all SCNT-Target/images for pseudo labels; report this as transductive.")
    parser.add_argument("--output-root", default="dataset/SCNT-Pseudo", help="Pseudo dataset output root.")
    parser.add_argument("--visual-dir", default="outputs/visualizations/pseudo_labels")
    parser.add_argument("--debug-conf-dir", default="outputs/debug_pseudo_labels_conf")
    parser.add_argument("--stats-csv", default="dataset/SCNT-Pseudo/pseudo_stats.csv")
    parser.add_argument("--outputs-stats-csv", default="outputs/pseudo_stats.csv")
    parser.add_argument("--link-mode", choices=["copy", "symlink", "hardlink", "auto"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--predict-conf", type=float, default=0.001, help="Low model confidence floor before class-specific filtering.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf-class-0", type=float, default=0.55)
    parser.add_argument("--conf-class-1", type=float, default=0.55)
    parser.add_argument("--conf-class-2", type=float, default=0.75)
    parser.add_argument("--max-width-height", type=float, default=0.95)
    parser.add_argument("--min-area", type=float, default=0.00001)
    parser.add_argument("--needle-position-prior", action="store_true", help="Relabel left-side needle predictions as holding and right-side needle predictions as injection.")
    parser.add_argument("--holding-max-x", type=float, default=0.45, help="With --needle-position-prior, needle boxes left of this x-center become class 1.")
    parser.add_argument("--injection-min-x", type=float, default=0.55, help="With --needle-position-prior, needle boxes right of this x-center become class 0.")
    return parser.parse_args()


def threshold_for_class(args: argparse.Namespace, class_id: int) -> float:
    return {
        0: args.conf_class_0,
        1: args.conf_class_1,
        2: args.conf_class_2,
    }.get(class_id, 1.0)


def apply_needle_position_prior(args: argparse.Namespace, class_id: int, x_center: float) -> int:
    if not args.needle_position_prior or class_id not in {0, 1}:
        return class_id
    if x_center <= args.holding_max_x:
        return 1
    if x_center >= args.injection_min_x:
        return 0
    return class_id


def draw_filtered_boxes(image_path: Path, kept: List[Tuple[int, float, float, float, float, float]], output_path: Path) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Missing dependency opencv-python. Install with: pip install -r requirements.txt") from exc

    img = cv2.imread(str(image_path))
    if img is None:
        return
    height, width = img.shape[:2]
    for cls, x, y, w, h, conf in kept:
        x1 = int((x - w / 2.0) * width)
        y1 = int((y - h / 2.0) * height)
        x2 = int((x + w / 2.0) * width)
        y2 = int((y + h / 2.0) * height)
        color = COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.2f}".format(CLASS_NAMES.get(cls, cls), conf)
        cv2.putText(img, label, (max(x1, 0), max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), img)


def filter_boxes(args: argparse.Namespace, boxes) -> Tuple[List[Tuple[int, float, float, float, float, float]], Dict[str, int]]:
    kept: List[Tuple[int, float, float, float, float, float]] = []
    filtered = Counter()
    if boxes is None or len(boxes) == 0:
        return kept, {"low_conf": 0, "invalid_size": 0, "invalid_class": 0, "total": 0}

    xywhn = boxes.xywhn.detach().cpu().numpy()
    cls_arr = boxes.cls.detach().cpu().numpy()
    conf_arr = boxes.conf.detach().cpu().numpy()

    for coords, cls_value, conf_value in zip(xywhn, cls_arr, conf_arr):
        cls = int(cls_value)
        conf = float(conf_value)
        x, y, w, h = [float(v) for v in coords]
        cls = apply_needle_position_prior(args, cls, x)
        if cls not in CLASS_NAMES:
            filtered["invalid_class"] += 1
            continue
        if conf < threshold_for_class(args, cls):
            filtered["low_conf"] += 1
            continue
        if w <= 0.0 or h <= 0.0 or w > args.max_width_height or h > args.max_width_height or (w * h) < args.min_area:
            filtered["invalid_size"] += 1
            continue
        kept.append((cls, x, y, w, h, conf))

    filtered["total"] = sum(filtered.values())
    return kept, dict(filtered)


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency ultralytics. Install with: pip install -r requirements.txt") from exc

    root = project_root()
    model_path = resolve_path(args.model, root)
    image_dir = resolve_path("dataset/SCNT/SCNT-Target/images", root) if args.transductive_all_target else resolve_path(args.image_dir, root)
    output_root = resolve_path(args.output_root, root)
    pseudo_images_dir = output_root / "images"
    pseudo_labels_dir = output_root / "labels"
    visual_dir = resolve_path(args.visual_dir, root)
    debug_dir = resolve_path(args.debug_conf_dir, root)
    stats_csv = resolve_path(args.stats_csv, root)
    outputs_stats_csv = resolve_path(args.outputs_stats_csv, root)

    if args.overwrite:
        for path in [output_root, visual_dir, debug_dir]:
            if path.exists():
                safe_rmtree(path)

    for path in [pseudo_images_dir, pseudo_labels_dir, visual_dir, debug_dir]:
        ensure_dir(path)

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError("No images found in {}. Run split_target.py first or pass --transductive-all-target.".format(image_dir))

    if args.transductive_all_target:
        print(
            "WARNING: Using all target images for pseudo labels. This is a transductive setting; "
            "validation on SCNT-Target/labels may be considered information-use controversial."
        )

    model = YOLO(str(model_path))
    predict_kwargs = {
        "source": [str(p) for p in images],
        "conf": args.predict_conf,
        "imgsz": args.imgsz,
        "stream": True,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    rows: List[Dict[str, object]] = []
    class_counts = Counter()
    conf_sums = defaultdict(float)

    for input_image_path, result in zip(images, model.predict(**predict_kwargs)):
        # Ultralytics may expose result.path as image0.jpg/image1.jpg when
        # source is a Python list, so keep the original input path as truth.
        image_path = input_image_path
        rel = image_path.relative_to(image_dir)
        kept, filtered = filter_boxes(args, result.boxes)

        link_or_copy_file(image_path, pseudo_images_dir / rel, args.link_mode)
        label_path = pseudo_labels_dir / rel.with_suffix(".txt")
        debug_label_path = debug_dir / rel.with_suffix(".txt")
        ensure_dir(label_path.parent)
        ensure_dir(debug_label_path.parent)

        with label_path.open("w", encoding="utf-8") as f:
            for cls, x, y, w, h, _conf in kept:
                f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(cls, x, y, w, h))
        with debug_label_path.open("w", encoding="utf-8") as f:
            for cls, x, y, w, h, conf in kept:
                f.write("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(cls, x, y, w, h, conf))

        draw_filtered_boxes(image_path, kept, visual_dir / rel)

        per_class = Counter(cls for cls, *_ in kept)
        for cls, *_rest, conf in kept:
            class_counts[cls] += 1
            conf_sums[cls] += conf
        mean_conf = sum(item[-1] for item in kept) / len(kept) if kept else 0.0
        row = {
            "image": rel.as_posix(),
            "pseudo_count": len(kept),
            "class_0_count": per_class.get(0, 0),
            "class_1_count": per_class.get(1, 0),
            "class_2_count": per_class.get(2, 0),
            "mean_conf": "{:.6f}".format(mean_conf),
            "filtered_low_conf": filtered.get("low_conf", 0),
            "filtered_invalid_size": filtered.get("invalid_size", 0),
            "filtered_invalid_class": filtered.get("invalid_class", 0),
            "filtered_total": filtered.get("total", 0),
        }
        rows.append(row)

    fieldnames = [
        "image",
        "pseudo_count",
        "class_0_count",
        "class_1_count",
        "class_2_count",
        "mean_conf",
        "filtered_low_conf",
        "filtered_invalid_size",
        "filtered_invalid_class",
        "filtered_total",
    ]
    for path in [stats_csv, outputs_stats_csv]:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    class_stats_path = output_root / "pseudo_class_stats.csv"
    with class_stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "count", "mean_conf"])
        writer.writeheader()
        for cls, name in CLASS_NAMES.items():
            count = class_counts.get(cls, 0)
            mean_conf = conf_sums[cls] / count if count else 0.0
            writer.writerow({"class_id": cls, "class_name": name, "count": count, "mean_conf": "{:.6f}".format(mean_conf)})

    print("Pseudo labels saved to {}".format(pseudo_labels_dir))
    print("Pseudo images saved to {}".format(pseudo_images_dir))
    print("Pseudo visualizations saved to {}".format(visual_dir))
    print("Stats saved to {} and {}".format(stats_csv, outputs_stats_csv))


if __name__ == "__main__":
    main()
