from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from common import CLASS_NAMES, ensure_dir, list_images, project_root, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-label selected few-shot images with a YOLO model.")
    parser.add_argument("--model", required=True, help="YOLO weights, e.g. runs/scnt/source_aug_xxx/weights/best.pt")
    parser.add_argument("--image-dir", default="dataset/SCNT-Fewshot/images")
    parser.add_argument("--label-dir", default="dataset/SCNT-Fewshot/labels")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--conf-class-0", type=float, default=0.35)
    parser.add_argument("--conf-class-1", type=float, default=0.35)
    parser.add_argument("--conf-class-2", type=float, default=0.50)
    parser.add_argument("--overwrite-labels", action="store_true", help="Overwrite existing .txt labels.")
    return parser.parse_args()


def threshold_for_class(args: argparse.Namespace, class_id: int) -> float:
    return {
        0: args.conf_class_0,
        1: args.conf_class_1,
        2: args.conf_class_2,
    }.get(class_id, 1.0)


def filter_boxes(args: argparse.Namespace, boxes) -> List[Tuple[int, float, float, float, float]]:
    kept: List[Tuple[int, float, float, float, float]] = []
    if boxes is None or len(boxes) == 0:
        return kept

    xywhn = boxes.xywhn.detach().cpu().numpy()
    cls_arr = boxes.cls.detach().cpu().numpy()
    conf_arr = boxes.conf.detach().cpu().numpy()

    for coords, cls_value, conf_value in zip(xywhn, cls_arr, conf_arr):
        cls = int(cls_value)
        conf = float(conf_value)
        if cls not in CLASS_NAMES or conf < threshold_for_class(args, cls):
            continue
        x, y, w, h = [float(v) for v in coords]
        if w <= 0.0 or h <= 0.0 or w > 0.95 or h > 0.95 or (w * h) < 0.00001:
            continue
        kept.append((cls, x, y, w, h))
    return kept


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency ultralytics. Install with: pip install -r requirements.txt") from exc

    root = project_root()
    model_path = resolve_path(args.model, root)
    image_dir = resolve_path(args.image_dir, root)
    label_dir = resolve_path(args.label_dir, root)
    ensure_dir(label_dir)

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError("No images found in {}".format(image_dir))

    model = YOLO(str(model_path))
    written = 0
    skipped = 0
    for image_path in images:
        rel = image_path.relative_to(image_dir)
        label_path = label_dir / rel.with_suffix(".txt")
        if label_path.exists() and not args.overwrite_labels:
            skipped += 1
            continue

        predict_kwargs = {
            "source": str(image_path),
            "conf": args.predict_conf,
            "imgsz": args.imgsz,
            "batch": 1,
            "stream": False,
            "verbose": False,
        }
        if args.device is not None:
            predict_kwargs["device"] = args.device
        result = model.predict(**predict_kwargs)[0]

        ensure_dir(label_path.parent)
        kept = filter_boxes(args, result.boxes)
        with label_path.open("w", encoding="utf-8") as f:
            for cls, x, y, w, h in kept:
                f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(cls, x, y, w, h))
        written += 1

    print("Pre-label done. labels_written={} labels_skipped_existing={}".format(written, skipped))
    print("Labels: {}".format(label_dir))


if __name__ == "__main__":
    main()
