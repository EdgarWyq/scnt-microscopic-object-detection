from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Set

from common import ensure_dir, list_images, project_root, resolve_path, yolo_label_path_for_image


RIGHT_KEYS = {ord("n"), ord("d"), 2555904}
LEFT_KEYS = {ord("p"), ord("a"), 2424832}
SELECT_KEYS = {ord(" "), ord("y"), ord("\r"), ord("\n")}
UNSELECT_KEYS = {ord("u"), ord("x"), 8, 127}
QUIT_KEYS = {ord("q"), 27}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rapidly review SCNT target visualizations and select images for manual annotation.")
    parser.add_argument("--image-dir", default="dataset/SCNT/SCNT-Target/images", help="Clean target images.")
    parser.add_argument(
        "--visual-dir",
        default="outputs/visualizations/yolo11s_all_target_raw_no_postprocess_all",
        help="Prediction visualization images used for review.",
    )
    parser.add_argument("--target-label-dir", default="dataset/SCNT/SCNT-Target/labels", help="Target GT labels, copied only to final_eval.")
    parser.add_argument("--output-root", default="outputs/manual_self_select")
    parser.add_argument("--export-root", default="dataset/SCNT-ManualSelf")
    parser.add_argument("--selection-file", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--target-count", type=int, default=120)
    parser.add_argument("--max-width", type=int, default=1500)
    parser.add_argument("--max-height", type=int, default=900)
    parser.add_argument("--reset-selection", action="store_true", help="Ignore the previous selected_images.txt and start from an empty selection.")
    parser.add_argument("--overwrite-export", action="store_true", help="Clear exported image/eval folders before copying.")
    parser.add_argument("--no-export", action="store_true", help="Only save selected_images.txt, do not copy split files.")
    return parser.parse_args()


def reset_dir(path: Path, overwrite: bool) -> None:
    root = project_root().resolve()
    target = path.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("Refusing to reset path outside project: {}".format(target))
    if overwrite and target.exists():
        shutil.rmtree(str(target))
    ensure_dir(target)


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))
    return True


def load_selection(selection_file: Path) -> Set[str]:
    if not selection_file.exists():
        return set()
    return {line.strip() for line in selection_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def save_selection(selection_file: Path, selected: Set[str], images: List[Path], csv_path: Path) -> None:
    ensure_dir(selection_file.parent)
    ordered = [image.name for image in images if image.name in selected]
    selection_file.write_text("\n".join(ordered) + ("\n" if ordered else ""), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "selected"])
        writer.writeheader()
        for image in images:
            writer.writerow({"image": image.name, "selected": int(image.name in selected)})


def fit_scale(width: int, height: int, max_width: int, max_height: int) -> float:
    return min(max_width / max(width, 1), max_height / max(height, 1), 1.0)


def draw_status(image, index: int, total: int, filename: str, selected: Set[str], target_count: int):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Missing dependency opencv-python. Install with: pip install -r requirements.txt") from exc

    chosen = filename in selected
    bar_h = 42
    canvas = np.zeros((image.shape[0] + bar_h, image.shape[1], 3), dtype=image.dtype)
    canvas[:bar_h, :, :] = (18, 18, 18)
    canvas[bar_h:, :, :] = image
    state = "SELECTED" if chosen else "not selected"
    text = "[{}/{}] {} | {} | selected={}/{} | Space/y select, n skip, p prev, u unselect, s save, q quit".format(
        index + 1,
        total,
        filename,
        state,
        len(selected),
        target_count,
    )
    color = (80, 255, 80) if chosen else (235, 235, 235)
    cv2.putText(canvas, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)
    return canvas


def export_split(
    args: argparse.Namespace,
    images: List[Path],
    selected: Set[str],
    visual_dir: Path,
    target_label_dir: Path,
    export_root: Path,
    output_root: Path,
) -> Dict[str, int]:
    manual_images = export_root / "manual_train" / "images"
    manual_labels = export_root / "manual_train" / "labels"
    final_images = export_root / "final_eval" / "images"
    final_labels = export_root / "final_eval" / "labels"
    selected_visuals = output_root / "selected_visualizations"

    reset_dir(manual_images, args.overwrite_export)
    reset_dir(manual_labels, args.overwrite_export)
    reset_dir(final_images, args.overwrite_export)
    reset_dir(final_labels, args.overwrite_export)
    reset_dir(selected_visuals, args.overwrite_export)

    counts = {
        "manual_train_images": 0,
        "selected_visualizations": 0,
        "final_eval_images": 0,
        "final_eval_labels": 0,
        "missing_final_eval_labels": 0,
    }

    image_dir = resolve_path(args.image_dir, project_root())
    for image in images:
        if image.name in selected:
            if copy_file(image, manual_images / image.name):
                counts["manual_train_images"] += 1
            if copy_file(visual_dir / image.name, selected_visuals / image.name):
                counts["selected_visualizations"] += 1
            continue

        if copy_file(image, final_images / image.name):
            counts["final_eval_images"] += 1
        label_src = yolo_label_path_for_image(image, image_dir, target_label_dir)
        if copy_file(label_src, final_labels / label_src.name):
            counts["final_eval_labels"] += 1
        else:
            counts["missing_final_eval_labels"] += 1
    return counts


def main() -> None:
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Missing dependency opencv-python. Install with: pip install -r requirements.txt") from exc

    root = project_root()
    image_dir = resolve_path(args.image_dir, root)
    visual_dir = resolve_path(args.visual_dir, root)
    target_label_dir = resolve_path(args.target_label_dir, root)
    output_root = resolve_path(args.output_root, root)
    export_root = resolve_path(args.export_root, root)
    selection_file = resolve_path(args.selection_file, root) if args.selection_file else output_root / "selected_images.txt"
    selection_csv = output_root / "selection.csv"

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError("No images found in {}".format(image_dir))
    ensure_dir(output_root)
    selected = set() if args.reset_selection else load_selection(selection_file)

    index = min(max(args.start, 0), len(images) - 1)
    window = "SCNT manual image selector"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        image_path = images[index]
        visual_path = visual_dir / image_path.name
        shown_path = visual_path if visual_path.exists() else image_path
        image = cv2.imread(str(shown_path))
        if image is None:
            raise FileNotFoundError("Could not read {}".format(shown_path))
        scale = fit_scale(image.shape[1], image.shape[0] + 42, args.max_width, args.max_height)
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        frame = draw_status(image, index, len(images), image_path.name, selected, args.target_count)
        cv2.imshow(window, frame)
        key = cv2.waitKeyEx(0)

        if key in QUIT_KEYS:
            break
        if key == ord("s"):
            save_selection(selection_file, selected, images, selection_csv)
            continue
        if key in SELECT_KEYS:
            selected.add(image_path.name)
            if index < len(images) - 1:
                index += 1
            continue
        if key in UNSELECT_KEYS:
            selected.discard(image_path.name)
            continue
        if key in RIGHT_KEYS:
            if index < len(images) - 1:
                index += 1
            continue
        if key in LEFT_KEYS:
            if index > 0:
                index -= 1
            continue

    save_selection(selection_file, selected, images, selection_csv)
    counts: Dict[str, int] = {}
    if not args.no_export:
        counts = export_split(args, images, selected, visual_dir, target_label_dir, export_root, output_root)
    cv2.destroyAllWindows()

    print("Selection saved: {}".format(selection_file))
    print("Selection CSV: {}".format(selection_csv))
    print("selected_count={}".format(len(selected)))
    if counts:
        print("Export root: {}".format(export_root))
        for key, value in counts.items():
            print("{}={}".format(key, value))


if __name__ == "__main__":
    main()
