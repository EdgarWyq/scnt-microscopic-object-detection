from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set

from common import ensure_dir, list_images, project_root, resolve_path, yolo_label_path_for_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select high-value SCNT target images for manual active-learning annotation without reading target labels."
    )
    parser.add_argument("--model", required=True, help="YOLO weights used to score target images.")
    parser.add_argument("--image-dir", default="dataset/SCNT/SCNT-Target/images")
    parser.add_argument("--target-label-dir", default="dataset/SCNT/SCNT-Target/labels", help="Only copied to final_eval labels.")
    parser.add_argument("--visual-dir", default="outputs/visualizations/yolo11s_all_target_raw_no_postprocess_all")
    parser.add_argument("--output-root", default="outputs/manual_selection")
    parser.add_argument("--split-root", default="dataset/SCNT-ManualActive")
    parser.add_argument("--count", type=int, default=120, help="Number of manual-train images to select.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=None)
    parser.add_argument("--predict-conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reset_dir(path: Path, overwrite: bool) -> None:
    root = project_root().resolve()
    target = path.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("Refusing to reset path outside project: {}".format(target))
    if target.exists() and overwrite:
        shutil.rmtree(str(target))
    ensure_dir(target)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))
    return True


def image_color_features(image_path: Path) -> Dict[str, float]:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Missing dependency opencv-python. Install with: pip install -r requirements.txt") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        return {"orange_score": 0.0, "gray_score": 0.0}
    b, g, r = cv2.mean(image)[:3]
    orange_score = max(0.0, (r - b) / 255.0) + max(0.0, (r - g) / 255.0) * 0.5
    gray_score = 1.0 - min(1.0, (abs(r - g) + abs(g - b) + abs(r - b)) / (3.0 * 64.0))
    return {"orange_score": orange_score, "gray_score": max(0.0, gray_score)}


def prediction_features(args: argparse.Namespace, model, image_path: Path) -> Dict[str, object]:
    result = model.predict(
        source=str(image_path),
        conf=args.predict_conf,
        imgsz=args.imgsz,
        batch=1,
        stream=False,
        verbose=False,
        device=args.device,
    )[0]

    boxes = result.boxes
    features: Dict[str, object] = {
        "num_boxes": 0,
        "num_injection": 0,
        "num_holding": 0,
        "num_oocyte": 0,
        "mean_conf": 0.0,
        "min_conf": 0.0,
        "uncertain_count": 0,
        "small_box_count": 0,
        "holding_like_injection_count": 0,
        "edge_box_count": 0,
        "prelabel_lines": [],
    }
    if boxes is None or len(boxes) == 0:
        return features

    xywhn = boxes.xywhn.detach().cpu().numpy()
    cls_arr = boxes.cls.detach().cpu().numpy()
    conf_arr = boxes.conf.detach().cpu().numpy()

    confs: List[float] = []
    prelabel_lines: List[str] = []
    for coords, cls_value, conf_value in zip(xywhn, cls_arr, conf_arr):
        cls = int(cls_value)
        conf = float(conf_value)
        x, y, w, h = [float(v) for v in coords]
        if cls not in {0, 1, 2} or w <= 0.0 or h <= 0.0 or w > 0.95 or h > 0.95 or (w * h) < 0.00001:
            continue

        confs.append(conf)
        features["num_boxes"] = int(features["num_boxes"]) + 1
        if cls == 0:
            features["num_injection"] = int(features["num_injection"]) + 1
        elif cls == 1:
            features["num_holding"] = int(features["num_holding"]) + 1
        elif cls == 2:
            features["num_oocyte"] = int(features["num_oocyte"]) + 1

        area = w * h
        aspect = w / max(h, 1e-9)
        if conf < 0.55 or (cls == 2 and conf < 0.75):
            features["uncertain_count"] = int(features["uncertain_count"]) + 1
        if area < 0.01:
            features["small_box_count"] = int(features["small_box_count"]) + 1
        if cls == 0 and h >= 0.08 and aspect <= 5.0 and area >= 0.03:
            features["holding_like_injection_count"] = int(features["holding_like_injection_count"]) + 1
        if x - w / 2.0 < 0.04 or y - h / 2.0 < 0.04 or x + w / 2.0 > 0.96 or y + h / 2.0 > 0.96:
            features["edge_box_count"] = int(features["edge_box_count"]) + 1
        prelabel_lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(cls, x, y, w, h))

    if confs:
        features["mean_conf"] = sum(confs) / len(confs)
        features["min_conf"] = min(confs)
    features["prelabel_lines"] = prelabel_lines
    return features


def reasons_for(row: Dict[str, object]) -> List[str]:
    reasons: List[str] = []
    if int(row["holding_like_injection_count"]) > 0:
        reasons.append("holding_suspect")
    if int(row["uncertain_count"]) >= 2 or float(row["mean_conf"]) < 0.55:
        reasons.append("low_conf_uncertain")
    if int(row["num_boxes"]) <= 1:
        reasons.append("few_detections")
    if float(row["orange_score"]) >= 0.18:
        reasons.append("orange_background")
    if int(row["small_box_count"]) >= 2:
        reasons.append("small_objects")
    if int(row["edge_box_count"]) > 0:
        reasons.append("edge_or_truncated")
    if not reasons:
        reasons.append("diversity")
    return reasons


def add_by_reason(selected: List[Dict[str, object]], rows: List[Dict[str, object]], used: Set[str], reason: str, quota: int) -> None:
    candidates = [r for r in rows if reason in str(r["reasons"]).split(";") and r["image"] not in used]
    for row in candidates[:quota]:
        selected.append(row)
        used.add(str(row["image"]))


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency ultralytics. Install with: pip install -r requirements.txt") from exc

    root = project_root()
    image_dir = resolve_path(args.image_dir, root)
    target_label_dir = resolve_path(args.target_label_dir, root)
    visual_dir = resolve_path(args.visual_dir, root)
    output_root = resolve_path(args.output_root, root)
    split_root = resolve_path(args.split_root, root)

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError("No images found in {}".format(image_dir))
    if args.count <= 0 or args.count >= len(images):
        raise ValueError("--count must be between 1 and {}".format(len(images) - 1))

    random.seed(args.seed)
    model = YOLO(str(resolve_path(args.model, root)))

    rows: List[Dict[str, object]] = []
    for image_path in images:
        row: Dict[str, object] = {
            "image": image_path.name,
            "image_path": str(image_path),
        }
        row.update(image_color_features(image_path))
        row.update(prediction_features(args, model, image_path))
        row["reasons"] = ";".join(reasons_for(row))
        row["score"] = (
            8.0 * int(row["holding_like_injection_count"])
            + 2.5 * int(row["uncertain_count"])
            + 4.0 * (1 if int(row["num_boxes"]) <= 1 else 0)
            + 1.5 * int(row["small_box_count"])
            + 2.0 * float(row["orange_score"])
            + 1.0 * int(row["edge_box_count"])
            + random.random() * 0.01
        )
        rows.append(row)

    rows.sort(key=lambda r: float(r["score"]), reverse=True)

    selected: List[Dict[str, object]] = []
    used: Set[str] = set()
    quotas = [
        ("holding_suspect", max(35, args.count // 3)),
        ("low_conf_uncertain", max(25, args.count // 4)),
        ("few_detections", max(15, args.count // 8)),
        ("orange_background", max(15, args.count // 8)),
        ("small_objects", max(15, args.count // 8)),
        ("edge_or_truncated", max(10, args.count // 12)),
    ]
    for reason, quota in quotas:
        add_by_reason(selected, rows, used, reason, quota)
        if len(selected) >= args.count:
            break
    for row in rows:
        if len(selected) >= args.count:
            break
        if row["image"] not in used:
            selected.append(row)
            used.add(str(row["image"]))

    selected = selected[: args.count]
    selected_names = {str(r["image"]) for r in selected}
    final_eval = [p for p in images if p.name not in selected_names]

    reset_dir(output_root / "top_visualizations", args.overwrite)
    reset_dir(output_root / "top_clean_images", args.overwrite)
    reset_dir(split_root / "manual_train" / "images", args.overwrite)
    reset_dir(split_root / "manual_train" / "labels", args.overwrite)
    reset_dir(split_root / "final_eval" / "images", args.overwrite)
    reset_dir(split_root / "final_eval" / "labels", args.overwrite)

    for row in selected:
        image_path = Path(str(row["image_path"]))
        copy_if_exists(image_path, output_root / "top_clean_images" / image_path.name)
        copy_if_exists(visual_dir / image_path.name, output_root / "top_visualizations" / image_path.name)
        copy_if_exists(image_path, split_root / "manual_train" / "images" / image_path.name)
        label_path = split_root / "manual_train" / "labels" / image_path.with_suffix(".txt").name
        label_path.write_text("\n".join(row["prelabel_lines"]) + ("\n" if row["prelabel_lines"] else ""), encoding="utf-8")

    copied_eval_labels = 0
    missing_eval_labels = 0
    for image_path in final_eval:
        copy_if_exists(image_path, split_root / "final_eval" / "images" / image_path.name)
        src_label = yolo_label_path_for_image(image_path, image_dir, target_label_dir)
        if copy_if_exists(src_label, split_root / "final_eval" / "labels" / src_label.name):
            copied_eval_labels += 1
        else:
            missing_eval_labels += 1

    csv_path = output_root / "manual_candidate_scores.csv"
    fieldnames = [
        "selected",
        "image",
        "score",
        "reasons",
        "num_boxes",
        "num_injection",
        "num_holding",
        "num_oocyte",
        "mean_conf",
        "min_conf",
        "uncertain_count",
        "small_box_count",
        "holding_like_injection_count",
        "edge_box_count",
        "orange_score",
        "gray_score",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("1" if key == "selected" and row["image"] in selected_names else row.get(key, "")) for key in fieldnames})

    selected_list = output_root / "selected_manual_train.txt"
    selected_list.write_text("\n".join(str(r["image"]) for r in selected) + "\n", encoding="utf-8")

    print("Manual candidate selection done.")
    print("manual_train images: {}".format(len(selected)))
    print("final_eval images: {}".format(len(final_eval)))
    print("final_eval labels copied: {} missing: {}".format(copied_eval_labels, missing_eval_labels))
    print("Review visualizations: {}".format(output_root / "top_visualizations"))
    print("Manual split: {}".format(split_root))
    print("Scores CSV: {}".format(csv_path))


if __name__ == "__main__":
    main()
