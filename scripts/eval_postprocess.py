from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from common import CLASS_NAMES, ensure_dir, list_images, project_root, resolve_path


Box = Tuple[float, float, float, float]
Prediction = Tuple[int, float, Box, Tuple[float, float, float, float]]
GroundTruth = Tuple[int, Box]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO predictions before/after SCNT post-processing.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-dir", default="dataset/SCNT/target_split/target_eval/images")
    parser.add_argument("--label-dir", default="dataset/SCNT/target_split/target_eval/labels")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rect", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--output-csv", default="outputs/postprocess_eval.csv")
    parser.add_argument("--needle-morphology-relabel", action="store_true")
    parser.add_argument("--injection-to-holding-min-h", type=float, default=0.08)
    parser.add_argument("--injection-to-holding-max-aspect", type=float, default=5.0)
    parser.add_argument("--injection-to-holding-min-area", type=float, default=0.03)
    parser.add_argument("--filter-oocyte-size", action="store_true")
    parser.add_argument("--max-oocyte-area", type=float, default=0.09)
    parser.add_argument("--max-oocyte-side", type=float, default=0.36)
    return parser.parse_args()


def label_path_for_image(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    return label_dir / image_path.relative_to(image_dir).with_suffix(".txt")


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Box:
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def load_ground_truth(image_path: Path, image_dir: Path, label_dir: Path) -> List[GroundTruth]:
    label_path = label_path_for_image(image_path, image_dir, label_dir)
    gts: List[GroundTruth] = []
    if not label_path.exists():
        return gts
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if cls in CLASS_NAMES:
            gts.append((cls, xywh_to_xyxy(x, y, w, h)))
    return gts


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-12)


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap(
    predictions: Dict[int, List[Tuple[int, float, Box]]],
    ground_truths: Dict[int, Dict[int, List[Box]]],
    iou_threshold: float,
) -> Dict[int, float]:
    ap_by_class: Dict[int, float] = {}
    for cls in CLASS_NAMES:
        preds = sorted(predictions.get(cls, []), key=lambda item: item[1], reverse=True)
        gt_for_class = ground_truths.get(cls, {})
        total_gt = sum(len(items) for items in gt_for_class.values())
        if total_gt == 0:
            ap_by_class[cls] = 0.0
            continue
        matched = {image_idx: np.zeros(len(items), dtype=bool) for image_idx, items in gt_for_class.items()}
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for pred_idx, (image_idx, _conf, pred_box) in enumerate(preds):
            gt_boxes = gt_for_class.get(image_idx, [])
            if not gt_boxes:
                fp[pred_idx] = 1.0
                continue
            ious = np.array([box_iou(pred_box, gt_box) for gt_box in gt_boxes], dtype=np.float32)
            best_idx = int(np.argmax(ious))
            if ious[best_idx] >= iou_threshold and not matched[image_idx][best_idx]:
                tp[pred_idx] = 1.0
                matched[image_idx][best_idx] = True
            else:
                fp[pred_idx] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(float(total_gt), 1e-12)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_by_class[cls] = average_precision(recalls, precisions)
    return ap_by_class


def summarize_metrics(
    predictions: Dict[int, List[Tuple[int, float, Box]]],
    ground_truths: Dict[int, Dict[int, List[Box]]],
) -> Dict[str, float]:
    ap50 = compute_ap(predictions, ground_truths, 0.50)
    thresholds = np.arange(0.50, 0.96, 0.05)
    ap_by_threshold = [compute_ap(predictions, ground_truths, float(t)) for t in thresholds]
    row: Dict[str, float] = {}
    for cls, name in CLASS_NAMES.items():
        row[f"ap50_{name}"] = ap50[cls]
        row[f"map50_95_{name}"] = float(np.mean([aps[cls] for aps in ap_by_threshold]))
    row["map50"] = float(np.mean([ap50[cls] for cls in CLASS_NAMES]))
    row["map50_95"] = float(np.mean([row[f"map50_95_{name}"] for name in CLASS_NAMES.values()]))
    return row


def postprocess_predictions(args: argparse.Namespace, preds: Iterable[Prediction]) -> Tuple[List[Prediction], Counter]:
    out: List[Prediction] = []
    stats = Counter()
    for cls, conf, xyxy, xywhn in preds:
        x, y, w, h = xywhn
        area = w * h
        aspect = w / max(h, 1e-12)
        new_cls = cls
        if (
            cls == 0
            and args.needle_morphology_relabel
            and h >= args.injection_to_holding_min_h
            and aspect <= args.injection_to_holding_max_aspect
            and area >= args.injection_to_holding_min_area
        ):
            new_cls = 1
            stats["relabeled_injection_to_holding"] += 1
        if new_cls == 2 and args.filter_oocyte_size:
            if area > args.max_oocyte_area or max(w, h) > args.max_oocyte_side:
                stats["filtered_oocyte_size"] += 1
                continue
        out.append((new_cls, conf, xyxy, xywhn))
    return out, stats


def collect_predictions(args: argparse.Namespace, images: List[Path]) -> Tuple[Dict[int, List[Prediction]], Counter]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency ultralytics. Install with: pip install -r requirements.txt") from exc

    model = YOLO(str(resolve_path(args.model, project_root())))
    by_image: Dict[int, List[Prediction]] = {}
    stats = Counter()
    for image_idx, image_path in enumerate(images):
        kwargs = {
            "source": str(image_path),
            "conf": args.predict_conf,
            "imgsz": args.imgsz,
            "batch": 1,
            "rect": args.rect,
            "stream": False,
            "verbose": False,
        }
        if args.device is not None:
            kwargs["device"] = args.device
        result = model.predict(**kwargs)[0]
        image_preds: List[Prediction] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxyn = boxes.xyxyn.detach().cpu().numpy()
            xywhn = boxes.xywhn.detach().cpu().numpy()
            cls_arr = boxes.cls.detach().cpu().numpy()
            conf_arr = boxes.conf.detach().cpu().numpy()
            for cls_value, conf_value, box_xyxy, box_xywhn in zip(cls_arr, conf_arr, xyxyn, xywhn):
                cls = int(cls_value)
                if cls not in CLASS_NAMES:
                    continue
                image_preds.append(
                    (
                        cls,
                        float(conf_value),
                        tuple(float(v) for v in box_xyxy),
                        tuple(float(v) for v in box_xywhn),
                    )
                )
        by_image[image_idx] = image_preds
        stats["raw_predictions"] += len(image_preds)
    return by_image, stats


def to_class_predictions(by_image: Dict[int, List[Prediction]]) -> Dict[int, List[Tuple[int, float, Box]]]:
    by_class: Dict[int, List[Tuple[int, float, Box]]] = defaultdict(list)
    for image_idx, preds in by_image.items():
        for cls, conf, xyxy, _xywhn in preds:
            by_class[cls].append((image_idx, conf, xyxy))
    return by_class


def matched_confusion(
    by_image: Dict[int, List[Prediction]],
    gts_by_image: Dict[int, List[GroundTruth]],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> Counter:
    counts = Counter()
    for image_idx, gts in gts_by_image.items():
        preds = [pred for pred in by_image.get(image_idx, []) if pred[1] >= conf_threshold]
        used = set()
        for gt_cls, gt_box in gts:
            best_iou = -1.0
            best_idx = None
            best_cls = None
            for pred_idx, (pred_cls, _conf, pred_box, _xywhn) in enumerate(preds):
                if pred_idx in used:
                    continue
                score = box_iou(gt_box, pred_box)
                if score > best_iou:
                    best_iou = score
                    best_idx = pred_idx
                    best_cls = pred_cls
            if best_iou >= iou_threshold and best_idx is not None:
                used.add(best_idx)
                counts[(gt_cls, best_cls)] += 1
            else:
                counts[(gt_cls, "miss")] += 1
    return counts


def print_metric_table(rows: List[Dict[str, object]]) -> None:
    header = ["mode", "AP50 injection", "AP50 holding", "AP50 oocyte", "mAP50", "mAP50-95"]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for row in rows:
        print(
            "{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(
                row["mode"],
                float(row["ap50_injection_needle"]),
                float(row["ap50_holding_needle"]),
                float(row["ap50_oocyte"]),
                float(row["map50"]),
                float(row["map50_95"]),
            )
        )


def main() -> None:
    args = parse_args()
    root = project_root()
    image_dir = resolve_path(args.image_dir, root)
    label_dir = resolve_path(args.label_dir, root)
    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError("No images found in {}".format(image_dir))

    gts_by_image: Dict[int, List[GroundTruth]] = {}
    gts_by_class: Dict[int, Dict[int, List[Box]]] = defaultdict(lambda: defaultdict(list))
    for image_idx, image_path in enumerate(images):
        gts = load_ground_truth(image_path, image_dir, label_dir)
        gts_by_image[image_idx] = gts
        for cls, box in gts:
            gts_by_class[cls][image_idx].append(box)

    raw_by_image, stats = collect_predictions(args, images)
    post_by_image: Dict[int, List[Prediction]] = {}
    post_stats = Counter()
    for image_idx, preds in raw_by_image.items():
        processed, item_stats = postprocess_predictions(args, preds)
        post_by_image[image_idx] = processed
        post_stats.update(item_stats)

    raw_row = {"mode": "raw", **summarize_metrics(to_class_predictions(raw_by_image), gts_by_class)}
    post_row = {"mode": "postprocess", **summarize_metrics(to_class_predictions(post_by_image), gts_by_class)}
    rows: List[Dict[str, object]] = [raw_row, post_row]

    output_csv = resolve_path(args.output_csv, root)
    ensure_dir(output_csv.parent)
    fieldnames = ["mode", "map50", "map50_95"]
    for class_name in CLASS_NAMES.values():
        fieldnames.extend([f"ap50_{class_name}", f"map50_95_{class_name}"])
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print_metric_table(rows)
    print("raw_predictions={}".format(stats["raw_predictions"]))
    print(
        "postprocess: relabeled_injection_to_holding={}, filtered_oocyte_size={}".format(
            post_stats["relabeled_injection_to_holding"],
            post_stats["filtered_oocyte_size"],
        )
    )
    for mode, by_image in [("raw", raw_by_image), ("postprocess", post_by_image)]:
        counts = matched_confusion(by_image, gts_by_image)
        total_holding = sum(value for (gt, _pred), value in counts.items() if gt == 1)
        print("\n{} holding matched confusion at conf>=0.25, IoU>=0.5:".format(mode))
        for pred in [0, 1, 2, "miss"]:
            value = counts[(1, pred)]
            label = pred if isinstance(pred, str) else CLASS_NAMES[pred]
            ratio = value / total_holding if total_holding else 0.0
            print("  true holding -> {}: {} ({:.3f})".format(label, value, ratio))
    print("\nSaved {}".format(output_csv))


if __name__ == "__main__":
    main()
