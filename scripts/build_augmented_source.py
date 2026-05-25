from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from common import ensure_dir, list_images, project_root, resolve_path, safe_rmtree, write_dataset_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline-augmented SCNT source dataset.")
    parser.add_argument("--source-root", default="dataset/SCNT/SCNT-Source")
    parser.add_argument("--output-root", default="dataset/SCNT-Source-Aug")
    parser.add_argument("--config-output", default="configs/scnt_source_aug.yaml")
    parser.add_argument("--val-images", default=None, help="Validation images. Defaults to target_eval/images if present.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Randomly keep this fraction of source images before augmentation. Useful when source images are repetitive.",
    )
    parser.add_argument(
        "--max-variants-per-image",
        type=int,
        default=None,
        help="Randomly keep at most this many variants per source image. The orig variant is kept when present.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--style-image-dir",
        default=None,
        help="Target-domain images used only for color statistics. Defaults to target_split/target_adapt/images.",
    )
    parser.add_argument("--style-max-images", type=int, default=160)
    parser.add_argument(
        "--variants",
        default="orig,gaussian_blur,motion_blur,color_shift,warm_red,blur_color",
        help=(
            "Comma-separated variants. Supported: orig, gaussian_blur, motion_blur, color_shift, "
            "warm_red, bright_orange, blur_color, zoom_out_075, zoom_out_060, zoom_out_red_060, "
            "zoom_out_030, zoom_out_orange_030, target_color, target_orange, target_orange_strong, "
            "zoom_out_target_orange_030."
        ),
    )
    return parser.parse_args()


def label_for_image(image: Path, images_dir: Path, labels_dir: Path) -> Path:
    return labels_dir / image.relative_to(images_dir).with_suffix(".txt")


def write_image(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    ensure_dir(path.parent)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        cv2.imwrite(str(path), image)


def gaussian_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice([3, 5, 7])
    sigma = rng.uniform(0.6, 1.8)
    return cv2.GaussianBlur(image, (k, k), sigma)


def motion_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice([5, 7, 9])
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= float(k)
    return cv2.filter2D(image, -1, kernel)


def color_shift(image: np.ndarray, rng: random.Random) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10.0, 10.0)) % 180.0
    hsv[..., 1] *= rng.uniform(0.65, 1.35)
    hsv[..., 2] *= rng.uniform(0.65, 1.35)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    alpha = rng.uniform(0.85, 1.25)
    beta = rng.uniform(-18.0, 18.0)
    shifted = shifted * alpha + beta
    gamma = rng.uniform(0.75, 1.35)
    shifted = 255.0 * np.power(np.clip(shifted, 0, 255) / 255.0, gamma)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def warm_red(image: np.ndarray, rng: random.Random) -> np.ndarray:
    # Simulate the bright orange microscope background seen in target-domain images.
    out = image.astype(np.float32)
    out[..., 2] *= rng.uniform(1.45, 1.85)  # red channel
    out[..., 1] *= rng.uniform(1.12, 1.35)  # green channel
    out[..., 0] *= rng.uniform(0.35, 0.62)  # blue channel

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    orange = np.zeros_like(out)
    orange[..., 2] = np.clip(130.0 + 0.62 * gray, 0, 255)
    orange[..., 1] = np.clip(58.0 + 0.48 * gray, 0, 255)
    orange[..., 0] = np.clip(8.0 + 0.12 * gray, 0, 255)

    height = image.shape[0]
    vertical = np.linspace(rng.uniform(1.12, 1.28), rng.uniform(0.82, 0.98), height, dtype=np.float32)
    out = cv2.addWeighted(out, rng.uniform(0.30, 0.45), orange, rng.uniform(0.55, 0.70), 0.0)
    out *= vertical[:, None, None]

    # Keep the needle/oocyte contrast while pushing the field toward a bright orange cast.
    out = out * rng.uniform(1.02, 1.18) + rng.uniform(6.0, 20.0)
    gamma = rng.uniform(0.78, 0.92)
    out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    return np.clip(out, 0, 255).astype(np.uint8)


def bright_orange(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = warm_red(image, rng)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = np.clip(hsv[..., 0] + rng.uniform(-2.0, 5.0), 6.0, 24.0)
    hsv[..., 1] *= rng.uniform(1.08, 1.28)
    hsv[..., 2] *= rng.uniform(1.03, 1.18)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def blur_color(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = color_shift(image, rng)
    if rng.random() < 0.5:
        return gaussian_blur(out, rng)
    return motion_blur(out, rng)


TRANSFORMS: Dict[str, Callable[[np.ndarray, random.Random], np.ndarray]] = {
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "color_shift": color_shift,
    "warm_red": warm_red,
    "bright_orange": bright_orange,
    "blur_color": blur_color,
}


def default_style_image_dir(root: Path) -> Path:
    strict_dir = root / "dataset" / "SCNT" / "target_split" / "target_adapt" / "images"
    if strict_dir.exists():
        return strict_dir
    return root / "dataset" / "SCNT" / "SCNT-Target" / "images"


def orange_score(image: np.ndarray) -> float:
    small = cv2.resize(image, (80, 60), interpolation=cv2.INTER_AREA)
    b, g, r = small.reshape(-1, 3).mean(axis=0)
    return float(max(r - b, 0.0) + 0.55 * max(g - b, 0.0) - 0.20 * abs(r - g))


def load_style_images(style_dir: Path, max_images: int, rng: random.Random) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    paths = list_images(style_dir)
    if not paths:
        raise FileNotFoundError("No style images found in {}".format(style_dir))
    rng.shuffle(paths)
    paths = paths[:max_images]
    loaded: List[Tuple[float, np.ndarray]] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        loaded.append((orange_score(image), image))
    if not loaded:
        raise FileNotFoundError("Could not read style images from {}".format(style_dir))

    all_images = [image for _score, image in loaded]
    ranked = sorted(loaded, key=lambda item: item[0], reverse=True)
    top_count = max(1, min(len(ranked), len(ranked) // 3 or 1))
    orange_images = [image for _score, image in ranked[:top_count]]
    return all_images, orange_images


def lab_color_transfer(source: np.ndarray, style: np.ndarray, strength: float, keep_l: float) -> np.ndarray:
    if style.shape[:2] != source.shape[:2]:
        style = cv2.resize(style, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)

    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    style_lab = cv2.cvtColor(style, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_lab = source_lab.copy()

    for channel in range(3):
        src = source_lab[..., channel]
        tgt = style_lab[..., channel]
        src_mean, src_std = float(src.mean()), float(src.std())
        tgt_mean, tgt_std = float(tgt.mean()), float(tgt.std())
        src_std = max(src_std, 1.0)
        transferred = (src - src_mean) * (tgt_std / src_std) + tgt_mean
        if channel == 0:
            transferred = keep_l * src + (1.0 - keep_l) * transferred
        out_lab[..., channel] = transferred

    transferred_bgr = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if strength >= 0.999:
        return transferred_bgr
    return cv2.addWeighted(transferred_bgr, strength, source, 1.0 - strength, 0.0)


def target_color_variant(
    variant: str,
    image: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    rng: random.Random,
    style_images: List[np.ndarray],
    orange_style_images: List[np.ndarray],
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    use_orange = "orange" in variant
    pool = orange_style_images if use_orange and orange_style_images else style_images
    style = rng.choice(pool)
    if variant == "target_color":
        colorized = lab_color_transfer(image, style, strength=0.88, keep_l=0.36)
        return colorized, labels
    if variant == "target_orange":
        colorized = lab_color_transfer(image, style, strength=0.94, keep_l=0.28)
        return colorized, labels
    if variant == "target_orange_strong":
        colorized = lab_color_transfer(image, style, strength=1.0, keep_l=0.16)
        return colorized, labels
    if variant == "zoom_out_target_orange_030":
        colorized = lab_color_transfer(image, style, strength=0.96, keep_l=0.22)
        return zoom_out_image_and_labels(colorized, labels, rng, 0.26, 0.34)
    raise ValueError("Unsupported target color variant: {}".format(variant))


TARGET_COLOR_VARIANTS = {"target_color", "target_orange", "target_orange_strong", "zoom_out_target_orange_030"}
HOLDING_GEOMETRY_VARIANTS = set()


def read_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    labels: List[Tuple[int, float, float, float, float]] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        labels.append((cls, x, y, w, h))
    return labels


def write_yolo_label(label_path: Path, labels: List[Tuple[int, float, float, float, float]]) -> None:
    ensure_dir(label_path.parent)
    lines = []
    for cls, x, y, w, h in labels:
        if w <= 0.0 or h <= 0.0:
            continue
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(cls, x, y, w, h))
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def norm_to_xyxy(label: Tuple[int, float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    _cls, x, y, w, h = label
    x1 = int(round((x - w / 2.0) * width))
    y1 = int(round((y - h / 2.0) * height))
    x2 = int(round((x + w / 2.0) * width))
    y2 = int(round((y + h / 2.0) * height))
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def xyxy_to_norm(cls: int, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, float, float, float, float]:
    x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    return (
        cls,
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        box_w / width,
        box_h / height,
    )


def box_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter / float(area_a + area_b - inter)


def fit_box_in_image(cx: float, cy: float, box_w: int, box_h: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(round(cx - box_w / 2.0))
    y1 = int(round(cy - box_h / 2.0))
    x2 = x1 + box_w
    y2 = y1 + box_h
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        x1 -= x2 - width
        x2 = width
    if y2 > height:
        y1 -= y2 - height
        y2 = height
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def fill_region_with_border_median(image: np.ndarray, xyxy: Tuple[int, int, int, int]) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = xyxy
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    image[:] = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)


def paste_with_soft_edges(image: np.ndarray, patch: np.ndarray, xyxy: Tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = xyxy
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)
    patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if target_w < 8 or target_h < 8:
        image[y1:y2, x1:x2] = patch
        return
    mask = np.ones((target_h, target_w), dtype=np.float32)
    edge = max(3, min(target_w, target_h) // 12)
    mask[:edge, :] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)[:, None]
    mask[-edge:, :] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)[:, None]
    mask[:, :edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)[None, :]
    mask[:, -edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)[None, :]
    roi = image[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1.0 - mask[..., None]) + patch.astype(np.float32) * mask[..., None]
    image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)


def holding_horizontal_variant(
    image: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    rng: random.Random,
    style_images: Optional[List[np.ndarray]] = None,
    orange_style_images: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    base = image
    if style_images is not None and orange_style_images is not None:
        pool = orange_style_images if orange_style_images else style_images
        base = lab_color_transfer(image, rng.choice(pool), strength=0.94, keep_l=0.24)

    out = base.copy()
    height, width = out.shape[:2]
    updated = list(labels)
    other_boxes = [
        norm_to_xyxy(label, width, height)
        for label in labels
        if label[0] != 1
    ]

    for idx, label in enumerate(labels):
        cls, _x, _y, _w, _h = label
        if cls != 1:
            continue
        x1, y1, x2, y2 = norm_to_xyxy(label, width, height)
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            continue
        crop = base[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE if rng.random() < 0.5 else cv2.ROTATE_90_COUNTERCLOCKWISE)

        old_w = x2 - x1
        old_h = y2 - y1
        target_w = int(round(old_h * rng.uniform(0.82, 1.05)))
        target_h = int(round(old_w * rng.uniform(0.72, 0.98)))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        candidate_centers_y = [cy, cy - target_h * 1.15, cy + target_h * 1.15]
        candidates = [
            fit_box_in_image(cx, candidate_y, target_w, target_h, width, height)
            for candidate_y in candidate_centers_y
        ]
        def candidate_score(box: Tuple[int, int, int, int]) -> float:
            _x1, by1, _x2, by2 = box
            box_cy = (by1 + by2) / 2.0
            overlap = sum(box_iou_xyxy(box, other) for other in other_boxes)
            distance = abs(box_cy - cy) / height
            return 0.75 * overlap + distance

        best_box = min(candidates, key=candidate_score)

        fill_region_with_border_median(out, (x1, y1, x2, y2))
        paste_with_soft_edges(out, rotated, best_box)
        updated[idx] = xyxy_to_norm(1, *best_box, width, height)

    return out, updated


def zoom_out_image_and_labels(
    image: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    rng: random.Random,
    scale_min: float,
    scale_max: float,
    red_shift: bool = False,
    orange_shift: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    if orange_shift:
        base = bright_orange(image, rng)
    elif red_shift:
        base = warm_red(image, rng)
    else:
        base = image
    height, width = base.shape[:2]
    scale = rng.uniform(scale_min, scale_max)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(base, (new_width, new_height), interpolation=cv2.INTER_AREA)

    median = np.median(base.reshape(-1, base.shape[-1]), axis=0)
    canvas = np.empty_like(base)
    canvas[:] = median.astype(np.uint8)
    noise = rng.uniform(2.0, 8.0)
    canvas = np.clip(
        canvas.astype(np.float32) + rng.normalvariate(0.0, noise),
        0,
        255,
    ).astype(np.uint8)

    max_x = width - new_width
    max_y = height - new_height
    offset_x = rng.randint(0, max_x) if max_x > 0 else 0
    offset_y = rng.randint(0, max_y) if max_y > 0 else 0
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized

    out_labels: List[Tuple[int, float, float, float, float]] = []
    for cls, x, y, w, h in labels:
        out_x = x * scale + offset_x / width
        out_y = y * scale + offset_y / height
        out_w = w * scale
        out_h = h * scale
        out_labels.append((cls, out_x, out_y, out_w, out_h))
    return canvas, out_labels


def zoom_variant(
    variant: str,
    image: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    rng: random.Random,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    if variant == "zoom_out_075":
        return zoom_out_image_and_labels(image, labels, rng, 0.70, 0.80)
    if variant == "zoom_out_060":
        return zoom_out_image_and_labels(image, labels, rng, 0.55, 0.65)
    if variant == "zoom_out_red_060":
        return zoom_out_image_and_labels(image, labels, rng, 0.55, 0.65, red_shift=True)
    if variant == "zoom_out_030":
        return zoom_out_image_and_labels(image, labels, rng, 0.26, 0.34)
    if variant == "zoom_out_orange_030":
        return zoom_out_image_and_labels(image, labels, rng, 0.26, 0.34, orange_shift=True)
    raise ValueError("Unsupported zoom variant: {}".format(variant))


ZOOM_VARIANTS = {"zoom_out_075", "zoom_out_060", "zoom_out_red_060", "zoom_out_030", "zoom_out_orange_030"}


def copy_label(label_path: Path, dst_label: Path) -> None:
    ensure_dir(dst_label.parent)
    if label_path.exists():
        shutil.copy2(str(label_path), str(dst_label))
    else:
        dst_label.write_text("", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    source_root = resolve_path(args.source_root, root)
    output_root = resolve_path(args.output_root, root)
    images_dir = source_root / "images"
    labels_dir = source_root / "labels"
    out_images = output_root / "images"
    out_labels = output_root / "labels"
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    unknown = [
        v for v in variants
        if (
            v != "orig"
            and v not in TRANSFORMS
            and v not in ZOOM_VARIANTS
            and v not in TARGET_COLOR_VARIANTS
            and v not in HOLDING_GEOMETRY_VARIANTS
        )
    ]
    if unknown:
        raise ValueError("Unsupported variants: {}".format(", ".join(unknown)))

    if args.overwrite and output_root.exists():
        safe_rmtree(output_root)
    ensure_dir(out_images)
    ensure_dir(out_labels)

    images = list_images(images_dir)
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise FileNotFoundError("No source images found in {}".format(images_dir))

    rng = random.Random(args.seed)
    source_count_before_sampling = len(images)
    if args.sample_frac <= 0.0 or args.sample_frac > 1.0:
        raise ValueError("--sample-frac must be in (0, 1].")
    if args.sample_frac < 1.0:
        sample_count = max(1, int(round(len(images) * args.sample_frac)))
        images = sorted(rng.sample(images, sample_count))

    rows: List[Dict[str, str]] = []
    style_images: List[np.ndarray] = []
    orange_style_images: List[np.ndarray] = []
    if any(variant in TARGET_COLOR_VARIANTS or variant == "holding_horizontal_target_orange" for variant in variants):
        style_dir = resolve_path(args.style_image_dir, root) if args.style_image_dir else default_style_image_dir(root)
        style_images, orange_style_images = load_style_images(style_dir, args.style_max_images, rng)
        print(
            "Loaded {} target style images from {}; using {} orange-biased style references.".format(
                len(style_images), style_dir, len(orange_style_images)
            )
        )

    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print("Skipping unreadable image: {}".format(image_path))
            continue
        rel = image_path.relative_to(images_dir)
        label_path = label_for_image(image_path, images_dir, labels_dir)
        labels = read_yolo_label(label_path)

        image_variants = variants
        if args.max_variants_per_image is not None and args.max_variants_per_image > 0 and len(variants) > args.max_variants_per_image:
            if "orig" in variants and args.max_variants_per_image == 1:
                image_variants = ["orig"]
            elif "orig" in variants:
                candidates = [variant for variant in variants if variant != "orig"]
                image_variants = ["orig"] + rng.sample(candidates, args.max_variants_per_image - 1)
            else:
                image_variants = rng.sample(variants, args.max_variants_per_image)

        for variant in image_variants:
            if variant == "orig":
                augmented = image
                transformed_labels = labels
                suffix = ""
            elif variant in ZOOM_VARIANTS:
                augmented, transformed_labels = zoom_variant(variant, image, labels, rng)
                suffix = "_{}".format(variant)
            elif variant in TARGET_COLOR_VARIANTS:
                augmented, transformed_labels = target_color_variant(
                    variant, image, labels, rng, style_images, orange_style_images
                )
                suffix = "_{}".format(variant)
            elif variant in HOLDING_GEOMETRY_VARIANTS:
                if variant == "holding_horizontal_target_orange":
                    augmented, transformed_labels = holding_horizontal_variant(
                        image, labels, rng, style_images, orange_style_images
                    )
                else:
                    augmented, transformed_labels = holding_horizontal_variant(image, labels, rng)
                suffix = "_{}".format(variant)
            else:
                augmented = TRANSFORMS[variant](image, rng)
                transformed_labels = labels
                suffix = "_{}".format(variant)

            dst_rel = rel.with_name("{}{}{}".format(rel.stem, suffix, rel.suffix))
            dst_image = out_images / dst_rel
            dst_label = (out_labels / dst_rel).with_suffix(".txt")
            write_image(dst_image, augmented, args.jpeg_quality)
            if variant in ZOOM_VARIANTS or variant in TARGET_COLOR_VARIANTS or variant in HOLDING_GEOMETRY_VARIANTS:
                write_yolo_label(dst_label, transformed_labels)
            else:
                copy_label(label_path, dst_label)
            rows.append(
                {
                    "source_image": rel.as_posix(),
                    "augmented_image": dst_rel.as_posix(),
                    "variant": variant,
                    "label": dst_label.relative_to(output_root).as_posix(),
                }
            )

        if index % 500 == 0:
            print("Processed {}/{} source images".format(index, len(images)))

    manifest = output_root / "augmentation_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_image", "augmented_image", "variant", "label"])
        writer.writeheader()
        writer.writerows(rows)

    if args.val_images:
        val_images = resolve_path(args.val_images, root)
    else:
        strict_val = root / "dataset" / "SCNT" / "target_split" / "target_eval" / "images"
        val_images = strict_val if strict_val.exists() else root / "dataset" / "SCNT" / "SCNT-Target" / "images"
    write_dataset_yaml(resolve_path(args.config_output, root), out_images, val_images, root)

    print("Built offline augmented source dataset at {}".format(output_root))
    print("source images before sampling: {}".format(source_count_before_sampling))
    print("source images used: {}".format(len(images)))
    print("candidate variants: {}".format(len(variants)))
    if args.max_variants_per_image:
        print("max variants per image: {}".format(args.max_variants_per_image))
    print("total augmented training images: {}".format(len(rows)))
    print("manifest: {}".format(manifest))
    print("config: {}".format(resolve_path(args.config_output, root)))


if __name__ == "__main__":
    main()
