from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List

from common import (
    ensure_dir,
    link_or_copy_file,
    list_images,
    project_root,
    resolve_path,
    safe_rmtree,
    write_dataset_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create strict target_adapt / target_eval split for SCNT target domain.")
    parser.add_argument("--target-root", default="dataset/SCNT/SCNT-Target", help="Target domain root with images and labels.")
    parser.add_argument("--output-root", default="dataset/SCNT/target_split", help="Output split root.")
    parser.add_argument("--adapt-ratio", type=float, default=0.7, help="Ratio of target images used for pseudo-label adaptation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split.")
    parser.add_argument("--link-mode", choices=["copy", "symlink", "hardlink", "auto"], default="copy", help="How to place files.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing split output before creating a new one.")
    parser.add_argument("--update-configs", action="store_true", default=True, help="Update configs to validate on target_eval/images.")
    parser.add_argument("--no-update-configs", dest="update_configs", action="store_false", help="Do not update dataset yaml files.")
    return parser.parse_args()


def label_for_image(image_path: Path, target_images: Path, target_labels: Path) -> Path:
    return target_labels / image_path.relative_to(target_images).with_suffix(".txt")


def main() -> None:
    args = parse_args()
    root = project_root()
    target_root = resolve_path(args.target_root, root)
    output_root = resolve_path(args.output_root, root)
    target_images = target_root / "images"
    target_labels = target_root / "labels"

    images = list_images(target_images)
    if not images:
        raise FileNotFoundError("No target images found in {}".format(target_images))
    if not 0.0 < args.adapt_ratio < 1.0:
        raise ValueError("--adapt-ratio must be between 0 and 1 for a strict split.")

    if output_root.exists() and args.overwrite:
        safe_rmtree(output_root)

    adapt_images_dir = output_root / "target_adapt" / "images"
    eval_images_dir = output_root / "target_eval" / "images"
    eval_labels_dir = output_root / "target_eval" / "labels"
    for path in [adapt_images_dir, eval_images_dir, eval_labels_dir]:
        ensure_dir(path)

    rng = random.Random(args.seed)
    valid_images = [p for p in images if label_for_image(p, target_images, target_labels).exists()]
    missing_label_images = [p for p in images if not label_for_image(p, target_images, target_labels).exists()]
    if not valid_images:
        raise FileNotFoundError("No target images with labels found in {}".format(target_labels))

    eval_count = int(len(images) * (1.0 - args.adapt_ratio))
    if len(valid_images) > 1:
        eval_count = min(max(eval_count, 1), len(valid_images) - 1)
    else:
        eval_count = 0

    shuffled_valid: List[Path] = list(valid_images)
    rng.shuffle(shuffled_valid)
    eval_set = set(shuffled_valid[:eval_count])
    adapt_set = set(images) - eval_set

    manifest_path = output_root / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "label_copied"])
        writer.writeheader()
        for image_path in sorted(images):
            rel = image_path.relative_to(target_images)
            if image_path in adapt_set:
                link_or_copy_file(image_path, adapt_images_dir / rel, args.link_mode)
                writer.writerow({"image": rel.as_posix(), "split": "target_adapt", "label_copied": "false"})
            else:
                link_or_copy_file(image_path, eval_images_dir / rel, args.link_mode)
                src_label = label_for_image(image_path, target_images, target_labels)
                copied = False
                if src_label.exists():
                    link_or_copy_file(src_label, eval_labels_dir / rel.with_suffix(".txt"), args.link_mode)
                    copied = True
                writer.writerow({"image": rel.as_posix(), "split": "target_eval", "label_copied": str(copied).lower()})

    if args.update_configs:
        write_dataset_yaml(
            root / "configs" / "scnt_source.yaml",
            root / "dataset" / "SCNT" / "SCNT-Source" / "images",
            eval_images_dir,
            root,
        )
        write_dataset_yaml(
            root / "configs" / "scnt_pseudo.yaml",
            root / "dataset" / "SCNT-Adapted" / "images",
            eval_images_dir,
            root,
        )

    print("Target split created at {}".format(output_root))
    print("target_adapt images: {}".format(len(adapt_set)))
    print("target_eval images: {}".format(len(eval_set)))
    if missing_label_images:
        print("Images without labels were kept out of target_eval: {}".format(len(missing_label_images)))
    print("Manifest: {}".format(manifest_path))
    if args.update_configs:
        print("Updated configs/scnt_source.yaml and configs/scnt_pseudo.yaml to use target_eval/images as val.")


if __name__ == "__main__":
    main()
