from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from common import CLASS_NAMES, ensure_dir, list_images, project_root, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save YOLO prediction visualizations for target-domain images.")
    parser.add_argument("--model", required=True, help="Path to YOLO weights.")
    parser.add_argument("--source", default="dataset/SCNT/SCNT-Target/images", help="Image directory.")
    parser.add_argument("--output", default="outputs/visualizations/baseline", help="Visualization output directory.")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum images to visualize. Use <=0 for all images.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rect", action="store_true", help="Use minimal rectangular padding during prediction.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Reduce Ultralytics console output.")
    parser.add_argument("--filter-oocyte-size", action="store_true", help="Drop oversized oocyte boxes during visualization/application inference.")
    parser.add_argument("--max-oocyte-area", type=float, default=0.09)
    parser.add_argument("--max-oocyte-side", type=float, default=0.36)
    parser.add_argument(
        "--needle-morphology-relabel",
        action="store_true",
        help="Relabel thick, low-aspect injection predictions as holding needles during visualization/application inference.",
    )
    parser.add_argument("--injection-to-holding-min-h", type=float, default=0.08)
    parser.add_argument("--injection-to-holding-max-aspect", type=float, default=5.0)
    parser.add_argument("--injection-to-holding-min-area", type=float, default=0.03)
    parser.add_argument("--filter-stats-csv", default=None)
    return parser.parse_args()


def draw_filtered_predictions(args: argparse.Namespace, model, images, output_dir) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Missing dependency opencv-python. Install with: pip install -r requirements.txt") from exc

    colors = {
        0: (255, 60, 40),
        1: (255, 220, 40),
        2: (255, 255, 255),
    }
    rows = []
    totals = Counter()
    for image_path in images:
        kwargs = {
            "source": str(image_path),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "batch": 1,
            "rect": args.rect,
            "stream": False,
            "verbose": not args.quiet,
        }
        if args.device is not None:
            kwargs["device"] = args.device
        result = model.predict(**kwargs)[0]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        kept = 0
        filtered = 0
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            xywhn = boxes.xywhn.detach().cpu().numpy()
            cls_arr = boxes.cls.detach().cpu().numpy()
            conf_arr = boxes.conf.detach().cpu().numpy()
            for coords_xyxy, coords_xywhn, cls_value, conf_value in zip(xyxy, xywhn, cls_arr, conf_arr):
                cls = int(cls_value)
                conf = float(conf_value)
                _x, _y, box_w, box_h = [float(v) for v in coords_xywhn]
                aspect = box_w / max(box_h, 1e-9)
                area = box_w * box_h
                if (
                    cls == 0
                    and args.needle_morphology_relabel
                    and box_h >= args.injection_to_holding_min_h
                    and aspect <= args.injection_to_holding_max_aspect
                    and area >= args.injection_to_holding_min_area
                ):
                    cls = 1
                    totals["relabeled_injection_to_holding"] += 1
                if cls == 2 and args.filter_oocyte_size:
                    if area > args.max_oocyte_area or max(box_w, box_h) > args.max_oocyte_side:
                        filtered += 1
                        totals["filtered_oocyte_size"] += 1
                        continue
                x1, y1, x2, y2 = [int(round(v)) for v in coords_xyxy]
                color = colors.get(cls, (80, 220, 80))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = "{} {:.2f}".format(CLASS_NAMES.get(cls, str(cls)), conf)
                cv2.putText(image, label, (max(x1, 0), max(y1 - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                kept += 1
                totals["kept"] += 1
        rel_name = Path(image_path).name
        cv2.imwrite(str(output_dir / rel_name), image)
        rows.append({"image": rel_name, "kept": kept, "filtered": filtered})

    if args.filter_stats_csv:
        stats_path = resolve_path(args.filter_stats_csv, project_root())
    else:
        stats_path = output_dir / "filter_stats.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "kept", "filtered"])
        writer.writeheader()
        writer.writerows(rows)
    print("Filtered visualizations saved to {}".format(output_dir))
    print(
        "Filter stats saved to {} (filtered_oocyte_size={}, relabeled_injection_to_holding={})".format(
            stats_path,
            totals["filtered_oocyte_size"],
            totals["relabeled_injection_to_holding"],
        )
    )


def save_raw_predictions(args: argparse.Namespace, model, images, output_dir) -> None:
    for image_path in images:
        kwargs = {
            "source": str(image_path),
            "save": True,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "batch": 1,
            "rect": args.rect,
            "project": str(output_dir.parent),
            "name": output_dir.name,
            "exist_ok": True,
            "stream": False,
            "verbose": not args.quiet,
        }
        if args.device is not None:
            kwargs["device"] = args.device
        model.predict(**kwargs)
    print("Saved visualizations to {}".format(output_dir))


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency ultralytics. Install with: pip install -r requirements.txt") from exc

    root = project_root()
    model_path = resolve_path(args.model, root)
    source_dir = resolve_path(args.source, root)
    output_dir = resolve_path(args.output, root)
    images = list_images(source_dir)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise FileNotFoundError("No images found in {}".format(source_dir))
    ensure_dir(output_dir)

    model = YOLO(str(model_path))
    kwargs = {
        "source": [str(p) for p in images],
        "save": True,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "rect": args.rect,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": True if args.exist_ok else True,
        "verbose": not args.quiet,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    if args.filter_oocyte_size or args.needle_morphology_relabel:
        draw_filtered_predictions(args, model, images, output_dir)
    else:
        save_raw_predictions(args, model, images, output_dir)


if __name__ == "__main__":
    main()
