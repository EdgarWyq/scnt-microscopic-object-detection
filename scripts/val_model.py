from __future__ import annotations

import argparse
from pathlib import Path

from common import CLASS_NAMES, append_csv_row, extract_yolo_metrics, project_root, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO model on the configured SCNT validation split.")
    parser.add_argument("--model", default="runs/scnt/baseline/weights/best.pt", help="Path to YOLO weights.")
    parser.add_argument("--data", default="configs/scnt_source.yaml", help="Dataset yaml with target-domain val images.")
    parser.add_argument("--experiment", default="baseline", help="Fallback experiment name written to summary CSV.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--rect", action="store_true", help="Use rectangular validation to reduce padding for 4:3 images.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/scnt")
    parser.add_argument("--name", default=None, help="Validation run name and experiment name. Overrides --experiment.")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Reduce Ultralytics console output.")
    parser.add_argument("--summary-csv", default="outputs/experiments_summary.csv")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    return parser.parse_args()


def run_validation(args: argparse.Namespace) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency ultralytics. Install with: pip install -r requirements.txt") from exc

    root = project_root()
    model_path = resolve_path(args.model, root)
    data_yaml = resolve_path(args.data, root)
    project = resolve_path(args.project, root)
    summary_csv = resolve_path(args.summary_csv, root)
    if not model_path.exists() and model_path.suffix == ".pt":
        raise FileNotFoundError("Model weights not found: {}".format(model_path))

    model = YOLO(str(model_path))
    experiment_name = args.name or args.experiment
    val_kwargs = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "rect": args.rect,
        "project": str(project),
        "name": args.name or "val_{}".format(experiment_name),
        "exist_ok": args.exist_ok,
        "conf": args.conf,
        "iou": args.iou,
        "plots": True,
        "verbose": not args.quiet,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device
    metrics = model.val(**val_kwargs)

    row = extract_yolo_metrics(metrics, experiment_name, model_path, data_yaml)
    append_csv_row(summary_csv, row)
    return row


def fmt(value) -> str:
    try:
        return "{:.4f}".format(float(value))
    except (TypeError, ValueError):
        return "NA"


def main() -> None:
    args = parse_args()
    row = run_validation(args)
    print("Validation summary for experiment={}".format(row.get("experiment", args.name or args.experiment)))
    for class_id, class_name in CLASS_NAMES.items():
        print("AP@0.5 {}: {}".format(class_name, fmt(row.get("ap50_{}".format(class_name)))))
    print("mAP@0.5: {}".format(fmt(row.get("map50"))))
    print("mAP@[0.5:0.95]: {}".format(fmt(row.get("map50_95"))))
    print("Appended metrics to {}".format(resolve_path(args.summary_csv, project_root())))


if __name__ == "__main__":
    main()
