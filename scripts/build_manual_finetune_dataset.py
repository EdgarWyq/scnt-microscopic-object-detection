from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List

from common import ensure_dir, list_images, project_root, safe_rmtree, write_dataset_yaml, yolo_label_path_for_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact supervised target-domain fine-tuning dataset from sampled source, sampled source augmentations, and manual target labels."
    )
    parser.add_argument("--source-images", default="dataset/SCNT/SCNT-Source/images")
    parser.add_argument("--source-labels", default="dataset/SCNT/SCNT-Source/labels")
    parser.add_argument("--source-aug-root", default="dataset/SCNT-Source-Aug")
    parser.add_argument("--manual-images", default="dataset/SCNT-ManualSelf/manual_train/images")
    parser.add_argument("--manual-labels", default="dataset/SCNT-ManualSelf/manual_train/labels")
    parser.add_argument("--val-images", default="dataset/SCNT-ManualSelf/final_eval/images")
    parser.add_argument("--output-root", default="dataset/SCNT-ManualMix")
    parser.add_argument("--config", default="configs/scnt_manual_mix.yaml")
    parser.add_argument("--source-frac", type=float, default=0.20)
    parser.add_argument("--aug-per-source", type=int, default=2)
    parser.add_argument(
        "--manual-repeat",
        type=int,
        default=4,
        help="Repeat manual target images to prevent 50 target samples being drowned by source-domain data. Use 1 for strict single-use.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_pair(image_src: Path, label_src: Path, image_dst: Path, label_dst: Path) -> bool:
    if not image_src.exists() or not label_src.exists():
        return False
    ensure_dir(image_dst.parent)
    ensure_dir(label_dst.parent)
    shutil.copy2(str(image_src), str(image_dst))
    shutil.copy2(str(label_src), str(label_dst))
    return True


def load_aug_manifest(manifest_path: Path) -> Dict[str, List[Dict[str, str]]]:
    by_source: Dict[str, List[Dict[str, str]]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row.get("source_image", "")
            if not source:
                continue
            by_source.setdefault(source, []).append(row)
    return by_source


def main() -> None:
    args = parse_args()
    if not (0.0 < args.source_frac <= 1.0):
        raise ValueError("--source-frac must be in (0, 1].")
    if args.aug_per_source < 0:
        raise ValueError("--aug-per-source must be >= 0.")
    if args.manual_repeat < 1:
        raise ValueError("--manual-repeat must be >= 1.")

    root = project_root()
    rng = random.Random(args.seed)

    source_images_dir = (root / args.source_images).resolve()
    source_labels_dir = (root / args.source_labels).resolve()
    source_aug_root = (root / args.source_aug_root).resolve()
    manual_images_dir = (root / args.manual_images).resolve()
    manual_labels_dir = (root / args.manual_labels).resolve()
    val_images_dir = (root / args.val_images).resolve()
    output_root = (root / args.output_root).resolve()
    config_path = (root / args.config).resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError("{} already exists. Pass --overwrite to rebuild.".format(output_root))
        safe_rmtree(output_root)
    ensure_dir(output_root)

    out_images = output_root / "images"
    out_labels = output_root / "labels"
    manifest_rows: List[Dict[str, str]] = []

    source_images = list_images(source_images_dir)
    if not source_images:
        raise FileNotFoundError("No source images found in {}".format(source_images_dir))
    sample_count = max(1, round(len(source_images) * args.source_frac))
    selected_source = sorted(rng.sample(source_images, sample_count))

    copied_source = 0
    missing_source = 0
    for image_path in selected_source:
        label_path = yolo_label_path_for_image(image_path, source_images_dir, source_labels_dir)
        image_dst = out_images / "source_orig" / image_path.name
        label_dst = out_labels / "source_orig" / label_path.name
        if copy_pair(image_path, label_path, image_dst, label_dst):
            copied_source += 1
            manifest_rows.append(
                {
                    "split": "train",
                    "subset": "source_orig",
                    "source_image": image_path.name,
                    "output_image": image_dst.relative_to(output_root).as_posix(),
                    "output_label": label_dst.relative_to(output_root).as_posix(),
                    "variant": "orig",
                }
            )
        else:
            missing_source += 1

    aug_manifest = load_aug_manifest(source_aug_root / "augmentation_manifest.csv")
    copied_aug = 0
    missing_aug = 0
    for image_path in selected_source:
        rows = [row for row in aug_manifest.get(image_path.name, []) if row.get("variant") != "orig"]
        rng.shuffle(rows)
        for row in rows[: args.aug_per_source]:
            aug_name = row["augmented_image"]
            aug_label_rel = row.get("label", "labels/{}".format(Path(aug_name).with_suffix(".txt").name))
            aug_image_path = source_aug_root / "images" / aug_name
            aug_label_path = source_aug_root / aug_label_rel
            image_dst = out_images / "source_aug" / aug_name
            label_dst = out_labels / "source_aug" / Path(aug_label_rel).name
            if copy_pair(aug_image_path, aug_label_path, image_dst, label_dst):
                copied_aug += 1
                manifest_rows.append(
                    {
                        "split": "train",
                        "subset": "source_aug",
                        "source_image": image_path.name,
                        "output_image": image_dst.relative_to(output_root).as_posix(),
                        "output_label": label_dst.relative_to(output_root).as_posix(),
                        "variant": row.get("variant", "aug"),
                    }
                )
            else:
                missing_aug += 1

    manual_images = list_images(manual_images_dir)
    if not manual_images:
        raise FileNotFoundError("No manual target images found in {}".format(manual_images_dir))
    copied_manual = 0
    missing_manual = 0
    for image_path in manual_images:
        label_path = yolo_label_path_for_image(image_path, manual_images_dir, manual_labels_dir)
        for repeat_idx in range(args.manual_repeat):
            if repeat_idx == 0:
                out_name = image_path.name
            else:
                out_name = "{}_manualrep{:02d}{}".format(image_path.stem, repeat_idx, image_path.suffix)
            image_dst = out_images / "manual_target" / out_name
            label_dst = out_labels / "manual_target" / Path(out_name).with_suffix(".txt").name
            if copy_pair(image_path, label_path, image_dst, label_dst):
                copied_manual += 1
                manifest_rows.append(
                    {
                        "split": "train",
                        "subset": "manual_target",
                        "source_image": image_path.name,
                        "output_image": image_dst.relative_to(output_root).as_posix(),
                        "output_label": label_dst.relative_to(output_root).as_posix(),
                        "variant": "manual_repeat_{}".format(repeat_idx),
                    }
                )
            else:
                missing_manual += 1

    manifest_path = output_root / "build_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["split", "subset", "source_image", "output_image", "output_label", "variant"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_dataset_yaml(config_path, out_images, val_images_dir, root)

    print("Built manual fine-tune dataset: {}".format(output_root))
    print("source_orig copied={} missing={}".format(copied_source, missing_source))
    print("source_aug copied={} missing={}".format(copied_aug, missing_aug))
    print("manual_target copied={} missing={} repeat={}".format(copied_manual, missing_manual, args.manual_repeat))
    print("train_images_total={}".format(copied_source + copied_aug + copied_manual))
    print("val_images={}".format(len(list_images(val_images_dir))))
    print("config={}".format(config_path))
    print("manifest={}".format(manifest_path))


if __name__ == "__main__":
    main()
