from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from common import ensure_dir, latest_rows_by_experiment, project_root, read_csv_rows, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run queued SCNT experiments: source-only, augmented-source, pseudo-label adaptation."
    )
    parser.add_argument("--device", default="0", help="Use 0 for GPU, cpu for CPU.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--rect", action="store_true", help="Use rectangular training/validation to reduce padding for 4:3 images.")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers. Use 0 or 2 if RAM usage is high.")
    parser.add_argument("--source-epochs", type=int, default=100)
    parser.add_argument("--aug-epochs", type=int, default=100)
    parser.add_argument("--pseudo-epochs", type=int, default=50)
    parser.add_argument("--use-offline-aug", action="store_true", help="Build and train source_aug on blurred/color-shifted source images.")
    parser.add_argument("--offline-aug-config", default="configs/scnt_source_aug.yaml")
    parser.add_argument("--offline-aug-root", default="dataset/SCNT-Source-Aug")
    parser.add_argument("--offline-aug-variants", default="orig,gaussian_blur,motion_blur,color_shift,warm_red,blur_color")
    parser.add_argument("--project", default="runs/scnt")
    parser.add_argument("--run-tag", default=None, help="Suffix for run names. Defaults to current timestamp.")
    parser.add_argument("--summary-csv", default="outputs/experiments_summary.csv")
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--metric", choices=["map50_95", "map50"], default="map50_95", help="Metric used when --pseudo-teacher best.")
    parser.add_argument("--pseudo-teacher", choices=["best", "source_plain", "source_aug"], default="best")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-source-plain", action="store_true")
    parser.add_argument("--skip-source-aug", action="store_true")
    parser.add_argument("--skip-pseudo", action="store_true")
    parser.add_argument("--transductive-all-target", action="store_true")
    parser.add_argument("--pseudo-conf-class-0", type=float, default=0.55)
    parser.add_argument("--pseudo-conf-class-1", type=float, default=0.55)
    parser.add_argument("--pseudo-conf-class-2", type=float, default=0.75)
    parser.add_argument("--quiet-output", action="store_true", default=True, help="Reduce training progress output.")
    parser.add_argument("--no-quiet-output", dest="quiet_output", action="store_false")
    parser.add_argument("--exist-ok", action="store_true", default=True)
    parser.add_argument("--no-exist-ok", dest="exist_ok", action="store_false")
    return parser.parse_args()


def check_cuda_if_needed(device: str) -> None:
    if device.lower() == "cpu":
        return
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed in this environment.") from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to PyTorch in this environment. "
            "Finish installing GPU PyTorch first, or run with --device cpu."
        )


def arg_value(cmd: List[str], flag: str, default: Optional[str] = None) -> Optional[str]:
    if flag not in cmd:
        return default
    idx = cmd.index(flag)
    if idx + 1 >= len(cmd):
        return default
    return cmd[idx + 1]


def training_results_csv(cmd: List[str], cwd: Path) -> Optional[Path]:
    script = next((Path(part).name for part in cmd if part.endswith(".py")), "")
    if script not in {"train_baseline.py", "train_pseudo_adapt.py"}:
        return None
    project = Path(arg_value(cmd, "--project", "runs/scnt") or "runs/scnt")
    name = arg_value(cmd, "--name", "train") or "train"
    if not project.is_absolute():
        project = cwd / project
    return project / name / "results.csv"


def compact_epoch_line(results_csv: Path, last_epoch: Optional[str]) -> Optional[Tuple[str, str]]:
    if not results_csv.exists():
        return None
    try:
        rows = read_csv_rows(results_csv)
    except Exception:
        return None
    if not rows:
        return None
    row = rows[-1]
    epoch = row.get("epoch", "").strip()
    if not epoch or epoch == last_epoch:
        return None

    def get_float(name: str) -> str:
        value = row.get(name, "")
        try:
            return "{:.4f}".format(float(value))
        except ValueError:
            return "-"

    label = results_csv.parent.name
    line = (
        "[{}] epoch {} | box {} cls {} | P {} R {} | mAP50 {} mAP50-95 {}"
    ).format(
        label,
        epoch,
        get_float("train/box_loss"),
        get_float("train/cls_loss"),
        get_float("metrics/precision(B)"),
        get_float("metrics/recall(B)"),
        get_float("metrics/mAP50(B)"),
        get_float("metrics/mAP50-95(B)"),
    )
    return epoch, line


def run_step(cmd: List[str], log_file: Path, cwd: Path, quiet_output: bool = False) -> None:
    line = "\n\n$ {}\n".format(" ".join(cmd))
    print(line)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(line)
        log.flush()
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        if quiet_output:
            env["YOLO_VERBOSE"] = "False"
            env["TQDM_DISABLE"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        out_queue: "queue.Queue[str]" = queue.Queue()

        def read_stdout() -> None:
            assert proc.stdout is not None
            for line_from_proc in proc.stdout:
                out_queue.put(line_from_proc)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()

        results_csv = training_results_csv(cmd, cwd)
        last_epoch: Optional[str] = None
        last_heartbeat = time.time()

        while proc.poll() is None:
            while True:
                try:
                    out_line = out_queue.get_nowait()
                except queue.Empty:
                    break
                log.write(out_line)
                if not quiet_output:
                    print(out_line, end="")

            if quiet_output and results_csv is not None:
                update = compact_epoch_line(results_csv, last_epoch)
                if update is not None:
                    last_epoch, progress = update
                    print(progress)
                    log.write(progress + "\n")
                    last_heartbeat = time.time()
                elif time.time() - last_heartbeat > 120:
                    heartbeat = "[{}] running... waiting for next epoch".format(results_csv.parent.name)
                    print(heartbeat)
                    log.write(heartbeat + "\n")
                    last_heartbeat = time.time()

            log.flush()
            time.sleep(2)

        reader.join(timeout=10)
        while True:
            try:
                out_line = out_queue.get_nowait()
            except queue.Empty:
                break
            log.write(out_line)
            if not quiet_output:
                print(out_line, end="")

        if quiet_output and results_csv is not None:
            update = compact_epoch_line(results_csv, last_epoch)
            if update is not None:
                _last_epoch, progress = update
                print(progress)
                log.write(progress + "\n")

        code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)


def best_teacher_from_summary(summary_csv: Path, metric: str, project: Path) -> Tuple[str, Path]:
    rows = latest_rows_by_experiment(read_csv_rows(summary_csv))
    candidates = []
    for exp in ["source_plain", "source_aug"]:
        row = rows.get(exp)
        if not row:
            continue
        try:
            score = float(row.get(metric, ""))
        except ValueError:
            continue
        candidates.append((score, exp))
    if candidates:
        _score, exp = max(candidates, key=lambda item: item[0])
        return exp, project / exp / "weights" / "best.pt"

    # Fallback for interrupted or manually prepared runs.
    for exp in ["source_aug", "source_plain"]:
        weight = project / exp / "weights" / "best.pt"
        if weight.exists():
            return exp, weight
    raise FileNotFoundError("No source teacher weights found under {}".format(project))


def teacher_path(args: argparse.Namespace, root: Path) -> Tuple[str, Path]:
    project = resolve_path(args.project, root)
    if args.pseudo_teacher == "best":
        return best_teacher_from_summary(resolve_path(args.summary_csv, root), args.metric, project)
    return args.pseudo_teacher, project / args.pseudo_teacher / "weights" / "best.pt"


def main() -> None:
    args = parse_args()
    root = project_root()
    check_cuda_if_needed(args.device)

    log_dir = ensure_dir(resolve_path(args.log_dir, root))
    log_file = log_dir / "full_pipeline_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    py = sys.executable
    project = resolve_path(args.project, root)
    summary_csv = resolve_path(args.summary_csv, root)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    source_plain_name = "source_plain_{}".format(run_tag)
    source_aug_name = "source_aug_{}".format(run_tag)
    pseudo_adapt_name = "pseudo_adapt_{}".format(run_tag)

    print("Full pipeline log: {}".format(log_file))
    print("Device: {} | batch: {} | imgsz: {}".format(args.device, args.batch, args.imgsz))
    print("Run names: {}, {}, {}".format(source_plain_name, source_aug_name, pseudo_adapt_name))

    common_train = [
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--device", args.device,
        "--workers", str(args.workers),
        "--project", str(project),
        "--exist-ok",
    ]
    if args.rect:
        common_train.append("--rect")

    if not args.skip_split:
        run_step([py, "scripts/split_target.py", "--overwrite"], log_file, root, args.quiet_output)
    if not args.skip_check:
        run_step([py, "scripts/check_dataset.py"], log_file, root, args.quiet_output)

    if args.use_offline_aug and not args.skip_source_aug:
        run_step(
            [
                py, "scripts/build_augmented_source.py",
                "--output-root", args.offline_aug_root,
                "--config-output", args.offline_aug_config,
                "--variants", args.offline_aug_variants,
                "--overwrite",
            ],
            log_file,
            root,
            args.quiet_output,
        )

    if not args.skip_source_plain:
        run_step(
            [
                py, "scripts/train_baseline.py",
                "--epochs", str(args.source_epochs),
                "--name", source_plain_name,
                "--hsv-h", "0.0",
                "--hsv-s", "0.0",
                "--hsv-v", "0.0",
                "--scale", "0.0",
                "--translate", "0.0",
                "--mosaic", "0.0",
                "--mixup", "0.0",
                "--fliplr", "0.0",
                "--flipud", "0.0",
            ] + (["--quiet"] if args.quiet_output else []) + common_train,
            log_file,
            root,
            args.quiet_output,
        )
        run_step(
            [
                py, "scripts/val_model.py",
                "--model", str(project / source_plain_name / "weights" / "best.pt"),
                "--data", "configs/scnt_source.yaml",
                "--experiment", source_plain_name,
                "--imgsz", str(args.imgsz),
                "--batch", str(args.batch),
                "--device", args.device,
                "--summary-csv", str(summary_csv),
                "--exist-ok",
            ] + (["--rect"] if args.rect else []) + (["--quiet"] if args.quiet_output else []),
            log_file,
            root,
            args.quiet_output,
        )

    if not args.skip_source_aug:
        run_step(
            [
                py, "scripts/train_baseline.py",
                "--data", args.offline_aug_config if args.use_offline_aug else "configs/scnt_source.yaml",
                "--epochs", str(args.aug_epochs),
                "--name", source_aug_name,
                "--hsv-h", "0.015",
                "--hsv-s", "0.7",
                "--hsv-v", "0.4",
                "--scale", "0.5",
                "--translate", "0.1",
                "--mosaic", "0.5",
                "--mixup", "0.1",
                "--fliplr", "0.0",
                "--flipud", "0.0",
            ] + (["--quiet"] if args.quiet_output else []) + common_train,
            log_file,
            root,
            args.quiet_output,
        )
        run_step(
            [
                py, "scripts/val_model.py",
                "--model", str(project / source_aug_name / "weights" / "best.pt"),
                "--data", "configs/scnt_source.yaml",
                "--experiment", source_aug_name,
                "--imgsz", str(args.imgsz),
                "--batch", str(args.batch),
                "--device", args.device,
                "--summary-csv", str(summary_csv),
                "--exist-ok",
            ] + (["--rect"] if args.rect else []) + (["--quiet"] if args.quiet_output else []),
            log_file,
            root,
            args.quiet_output,
        )

    if not args.skip_pseudo:
        if args.pseudo_teacher == "best":
            candidates = [
                (source_plain_name, project / source_plain_name / "weights" / "best.pt"),
                (source_aug_name, project / source_aug_name / "weights" / "best.pt"),
            ]
            # The validation CSV uses timestamped experiment names in tagged runs. Pick the latest row by metric.
            rows = latest_rows_by_experiment(read_csv_rows(summary_csv))
            scored = []
            for exp, weight in candidates:
                row = rows.get(exp)
                try:
                    score = float(row.get(args.metric, "")) if row else float("-inf")
                except ValueError:
                    score = float("-inf")
                if weight.exists():
                    scored.append((score, exp, weight))
            if not scored:
                raise FileNotFoundError("No teacher weights found for this run tag.")
            _score, teacher_exp, teacher = max(scored, key=lambda item: item[0])
        else:
            if args.pseudo_teacher == "source_plain":
                teacher_exp, teacher = source_plain_name, project / source_plain_name / "weights" / "best.pt"
            elif args.pseudo_teacher == "source_aug":
                teacher_exp, teacher = source_aug_name, project / source_aug_name / "weights" / "best.pt"
            else:
                teacher_exp, teacher = teacher_path(args, root)
        if not teacher.exists():
            raise FileNotFoundError("Teacher weights not found: {}".format(teacher))
        print("Pseudo-label teacher: {} ({})".format(teacher_exp, teacher))

        pseudo_cmd = [
            py, "scripts/generate_pseudo_labels.py",
            "--model", str(teacher),
            "--overwrite",
            "--device", args.device,
            "--conf-class-0", str(args.pseudo_conf_class_0),
            "--conf-class-1", str(args.pseudo_conf_class_1),
            "--conf-class-2", str(args.pseudo_conf_class_2),
        ]
        if args.transductive_all_target:
            pseudo_cmd.append("--transductive-all-target")
        run_step(pseudo_cmd, log_file, root, args.quiet_output)
        run_step([py, "scripts/build_pseudo_dataset.py", "--overwrite"], log_file, root, args.quiet_output)
        run_step(
            [
                py, "scripts/train_pseudo_adapt.py",
                "--model", str(teacher),
                "--baseline-model", str(teacher),
                "--epochs", str(args.pseudo_epochs),
                "--name", pseudo_adapt_name,
                "--skip-baseline-val",
                "--imgsz", str(args.imgsz),
                "--batch", str(args.batch),
                "--device", args.device,
                "--workers", str(args.workers),
                "--project", str(project),
                "--summary-csv", str(summary_csv),
                "--exist-ok",
            ] + (["--rect"] if args.rect else []) + (["--quiet"] if args.quiet_output else []),
            log_file,
            root,
            args.quiet_output,
        )

    run_step([py, "scripts/compare_results.py", "--summary-csv", str(summary_csv)], log_file, root, args.quiet_output)
    print("All requested experiments finished. Log: {}".format(log_file))


if __name__ == "__main__":
    main()
