from __future__ import annotations

import argparse
from pathlib import Path

from common import copy_tree_flat_by_relative, ensure_dir, list_images, project_root, resolve_path, safe_rmtree, write_dataset_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build second-stage source + pseudo-target training dataset.")
    parser.add_argument("--source-root", default="dataset/SCNT/SCNT-Source")
    parser.add_argument("--pseudo-root", default="dataset/SCNT-Pseudo")
    parser.add_argument("--output-root", default="dataset/SCNT-Adapted")
    parser.add_argument("--val-images", default=None, help="Validation images. Defaults to target_eval/images if present, else SCNT-Target/images.")
    parser.add_argument("--config-output", default="configs/scnt_pseudo.yaml")
    parser.add_argument("--link-mode", choices=["copy", "symlink", "hardlink", "auto"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_label_files(labels_dir: Path):
    if not labels_dir.exists():
        return []
    return sorted(p for p in labels_dir.rglob("*.txt") if p.name.lower() != "classes.txt")


def main() -> None:
    args = parse_args()
    root = project_root()
    source_root = resolve_path(args.source_root, root)
    pseudo_root = resolve_path(args.pseudo_root, root)
    output_root = resolve_path(args.output_root, root)

    source_images_dir = source_root / "images"
    source_labels_dir = source_root / "labels"
    pseudo_images_dir = pseudo_root / "images"
    pseudo_labels_dir = pseudo_root / "labels"

    forbidden = (root / "dataset" / "SCNT" / "SCNT-Target" / "labels").resolve()
    try:
        pseudo_labels_dir.resolve().relative_to(forbidden)
        raise ValueError("Refusing to use SCNT-Target/labels as pseudo labels.")
    except ValueError as exc:
        if "Refusing" in str(exc):
            raise

    if args.overwrite and output_root.exists():
        safe_rmtree(output_root)

    dst_source_images = output_root / "images" / "source"
    dst_source_labels = output_root / "labels" / "source"
    dst_pseudo_images = output_root / "images" / "pseudo_target"
    dst_pseudo_labels = output_root / "labels" / "pseudo_target"
    for path in [dst_source_images, dst_source_labels, dst_pseudo_images, dst_pseudo_labels]:
        ensure_dir(path)

    source_images = list_images(source_images_dir)
    source_labels = list_label_files(source_labels_dir)
    pseudo_images = list_images(pseudo_images_dir)
    pseudo_labels = list_label_files(pseudo_labels_dir)
    if not source_images or not source_labels:
        raise FileNotFoundError("Source images/labels are missing under {}".format(source_root))
    if not pseudo_images or not pseudo_labels:
        raise FileNotFoundError("Pseudo images/labels are missing under {}. Run generate_pseudo_labels.py first.".format(pseudo_root))

    source_img_counts = copy_tree_flat_by_relative(source_images, source_images_dir, dst_source_images, args.link_mode)
    source_lbl_counts = copy_tree_flat_by_relative(source_labels, source_labels_dir, dst_source_labels, args.link_mode)
    pseudo_img_counts = copy_tree_flat_by_relative(pseudo_images, pseudo_images_dir, dst_pseudo_images, args.link_mode)
    pseudo_lbl_counts = copy_tree_flat_by_relative(pseudo_labels, pseudo_labels_dir, dst_pseudo_labels, args.link_mode)

    if args.val_images:
        val_images = resolve_path(args.val_images, root)
    else:
        strict_val = root / "dataset" / "SCNT" / "target_split" / "target_eval" / "images"
        val_images = strict_val if strict_val.exists() else root / "dataset" / "SCNT" / "SCNT-Target" / "images"

    config_output = resolve_path(args.config_output, root)
    write_dataset_yaml(config_output, output_root / "images", val_images, root)

    print("Built adapted dataset at {}".format(output_root))
    print("source images: {} | source labels: {}".format(len(source_images), len(source_labels)))
    print("pseudo images: {} | pseudo labels: {}".format(len(pseudo_images), len(pseudo_labels)))
    print("placement counts:")
    print("  source images {}".format(source_img_counts))
    print("  source labels {}".format(source_lbl_counts))
    print("  pseudo images {}".format(pseudo_img_counts))
    print("  pseudo labels {}".format(pseudo_lbl_counts))
    print("Updated {}".format(config_output))


if __name__ == "__main__":
    main()

