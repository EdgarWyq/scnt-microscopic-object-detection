from __future__ import annotations

import argparse
from pathlib import Path

from common import project_root, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train source-only YOLO baseline on SCNT-Source.")
    parser.add_argument("--data", default="configs/scnt_source.yaml", help="Ultralytics dataset yaml.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO checkpoint or model name.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--rect", action="store_true", help="Use rectangular training to reduce padding for 4:3 images.")
    parser.add_argument("--device", default=None, help="CUDA device like 0, or cpu. Default lets Ultralytics choose.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/scnt")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Reduce Ultralytics console output.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--cache", action="store_true", help="Cache images during training.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate. Leave unset for Ultralytics default.")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning-rate fraction. Leave unset for Ultralytics default.")
    parser.add_argument("--close-mosaic", type=int, default=None, help="Disable mosaic in the last N epochs.")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze first N layers during fine-tuning.")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="Ultralytics HSV hue augmentation.")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="Ultralytics HSV saturation augmentation.")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="Ultralytics HSV value augmentation.")
    parser.add_argument("--brightness", type=float, default=None, help="Recorded only; use hsv_v or offline augmentation for brightness.")
    parser.add_argument("--contrast", type=float, default=None, help="Recorded only; use offline augmentation for contrast.")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency ultralytics. Install with: pip install -r requirements.txt") from exc

    root = project_root()
    data_yaml = resolve_path(args.data, root)
    project = resolve_path(args.project, root)

    if args.brightness is not None or args.contrast is not None:
        print(
            "Note: Ultralytics YOLO does not expose generic brightness/contrast knobs in all versions. "
            "This script passes hsv_v, scale, translate, mosaic, mixup and flip settings; see README for offline blur/noise/contrast augmentation."
        )

    model = YOLO(args.model)
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
        "flipud": args.flipud,
        "plots": True,
        "verbose": not args.quiet,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.close_mosaic is not None:
        train_kwargs["close_mosaic"] = args.close_mosaic
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    print("Training baseline with data={}".format(data_yaml))
    print("Outputs will be saved under {}".format(project / args.name))
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
