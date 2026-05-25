from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import project_root, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue training from baseline on source + pseudo-target labels.")
    parser.add_argument("--model", default="runs/scnt/baseline/weights/best.pt", help="Baseline best.pt used to initialize adaptation.")
    parser.add_argument("--data", default="configs/scnt_pseudo.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--rect", action="store_true", help="Use rectangular training to reduce padding for 4:3 images.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/scnt")
    parser.add_argument("--name", default="pseudo_adapt")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Reduce Ultralytics console output.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--baseline-data", default="configs/scnt_source.yaml")
    parser.add_argument("--baseline-model", default="runs/scnt/baseline/weights/best.pt")
    parser.add_argument("--skip-baseline-val", action="store_true")
    parser.add_argument("--summary-csv", default="outputs/experiments_summary.csv")
    return parser.parse_args()


def call_val(model_path: Path, data_yaml: Path, experiment: str, args: argparse.Namespace) -> None:
    root = project_root()
    cmd = [
        sys.executable,
        str(root / "scripts" / "val_model.py"),
        "--model",
        str(model_path),
        "--data",
        str(data_yaml),
        "--experiment",
        experiment,
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--summary-csv",
        str(resolve_path(args.summary_csv, root)),
        "--exist-ok",
    ]
    if args.rect:
        cmd.append("--rect")
    if args.device is not None:
        cmd.extend(["--device", args.device])
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency ultralytics. Install with: pip install -r requirements.txt") from exc

    root = project_root()
    model_path = resolve_path(args.model, root)
    data_yaml = resolve_path(args.data, root)
    project = resolve_path(args.project, root)
    if not model_path.exists():
        raise FileNotFoundError("Baseline weights not found: {}. Train baseline first.".format(model_path))

    model = YOLO(str(model_path))
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "rect": args.rect,
        "workers": args.workers,
        "project": str(project),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "seed": args.seed,
        "patience": args.patience,
        "cache": args.cache,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "scale": args.scale,
        "translate": args.translate,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "fliplr": args.fliplr,
        "plots": True,
        "verbose": not args.quiet,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    print("Training pseudo-adaptation model with data={}".format(data_yaml))
    model.train(**train_kwargs)

    best_path = project / args.name / "weights" / "best.pt"
    if not args.skip_baseline_val:
        baseline_model = resolve_path(args.baseline_model, root)
        baseline_data = resolve_path(args.baseline_data, root)
        if baseline_model.exists():
            call_val(baseline_model, baseline_data, "baseline", args)
        else:
            print("Skipping baseline validation because weights were not found: {}".format(baseline_model))
    call_val(best_path, data_yaml, "pseudo_adapt", args)


if __name__ == "__main__":
    main()
