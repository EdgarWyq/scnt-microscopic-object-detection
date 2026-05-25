from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from common import CLASS_NAMES, ensure_dir, list_images, project_root, resolve_path


COLORS = {
    0: (40, 200, 255),
    1: (255, 120, 40),
    2: (80, 220, 80),
}


@dataclass
class Box:
    cls: int
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized(self, width: int, height: int) -> Tuple[float, float, float, float]:
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        x_center = ((x1 + x2) / 2.0) / width
        y_center = ((y1 + y2) / 2.0) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        return x_center, y_center, w, h

    def contains(self, x: int, y: int) -> bool:
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        return x1 <= x <= x2 and y1 <= y <= y2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small OpenCV YOLO annotator for SCNT few-shot labels.")
    parser.add_argument("--image-dir", default="dataset/SCNT-Fewshot/images")
    parser.add_argument("--label-dir", default="dataset/SCNT-Fewshot/labels")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=850)
    return parser.parse_args()


def load_boxes(label_path: Path, width: int, height: int) -> List[Box]:
    boxes: List[Box] = []
    if not label_path.exists():
        return boxes
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if cls not in CLASS_NAMES:
            continue
        x1 = int(round((x - w / 2.0) * width))
        y1 = int(round((y - h / 2.0) * height))
        x2 = int(round((x + w / 2.0) * width))
        y2 = int(round((y + h / 2.0) * height))
        boxes.append(Box(cls, x1, y1, x2, y2))
    return boxes


def save_boxes(label_path: Path, boxes: List[Box], width: int, height: int) -> None:
    ensure_dir(label_path.parent)
    lines = []
    for box in boxes:
        x, y, w, h = box.normalized(width, height)
        if w <= 0.0 or h <= 0.0:
            continue
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(box.cls, x, y, w, h))
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def fit_scale(width: int, height: int, max_width: int, max_height: int) -> float:
    return min(max_width / width, max_height / height, 1.0)


def to_display_point(x: int, y: int, scale: float) -> Tuple[int, int]:
    return int(round(x * scale)), int(round(y * scale))


def to_image_point(x: int, y: int, scale: float, width: int, height: int) -> Tuple[int, int]:
    ix = int(round(x / scale))
    iy = int(round(y / scale))
    return min(max(ix, 0), width - 1), min(max(iy, 0), height - 1)


def draw_view(image, boxes: List[Box], selected: Optional[int], active_cls: int, index: int, total: int, image_path: Path, scale: float):
    import cv2

    if scale != 1.0:
        shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        shown = image.copy()

    for i, box in enumerate(boxes):
        color = COLORS.get(box.cls, (255, 255, 255))
        thickness = 3 if i == selected else 2
        x1, y1 = to_display_point(box.x1, box.y1, scale)
        x2, y2 = to_display_point(box.x2, box.y2, scale)
        cv2.rectangle(shown, (x1, y1), (x2, y2), color, thickness)
        label = "{}:{}".format(box.cls, CLASS_NAMES[box.cls])
        cv2.putText(shown, label, (max(x1, 0), max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    status = "[{}/{}] {} | active={} {} | 0/1/2 class, drag add, click select, Del/right-click delete, n next, p prev, s save, q quit".format(
        index + 1,
        total,
        image_path.name,
        active_cls,
        CLASS_NAMES[active_cls],
    )
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(shown, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
    return shown


def main() -> None:
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Missing dependency opencv-python. Install with: pip install -r requirements.txt") from exc

    root = project_root()
    image_dir = resolve_path(args.image_dir, root)
    label_dir = resolve_path(args.label_dir, root)
    ensure_dir(label_dir)
    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError("No images found in {}".format(image_dir))

    index = min(max(args.start, 0), len(images) - 1)
    active_cls = 0
    selected: Optional[int] = None
    dragging = False
    drag_start: Optional[Tuple[int, int]] = None
    drag_current: Optional[Tuple[int, int]] = None
    boxes: List[Box] = []
    image = None
    image_path = images[index]
    label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
    width = height = 0
    scale = 1.0
    dirty = False

    window = "SCNT YOLO Annotator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def load_current() -> None:
        nonlocal image, image_path, label_path, width, height, scale, boxes, selected, dirty
        image_path = images[index]
        label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError("Could not read image {}".format(image_path))
        height, width = image.shape[:2]
        scale = fit_scale(width, height, args.max_width, args.max_height)
        boxes = load_boxes(label_path, width, height)
        selected = None
        dirty = False

    def save_current() -> None:
        nonlocal dirty
        save_boxes(label_path, boxes, width, height)
        dirty = False

    def mouse_callback(event, x, y, _flags, _param) -> None:
        nonlocal dragging, drag_start, drag_current, selected, dirty
        ix, iy = to_image_point(x, y, scale, width, height)
        if event == cv2.EVENT_LBUTTONDOWN:
            selected = None
            for i in range(len(boxes) - 1, -1, -1):
                if boxes[i].contains(ix, iy):
                    selected = i
                    return
            dragging = True
            drag_start = (ix, iy)
            drag_current = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            drag_current = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            if drag_start is not None:
                x1, y1 = drag_start
                x2, y2 = ix, iy
                if abs(x2 - x1) >= 4 and abs(y2 - y1) >= 4:
                    boxes.append(Box(active_cls, x1, y1, x2, y2))
                    selected = len(boxes) - 1
                    dirty = True
            drag_start = None
            drag_current = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i in range(len(boxes) - 1, -1, -1):
                if boxes[i].contains(ix, iy):
                    del boxes[i]
                    selected = None
                    dirty = True
                    break

    load_current()
    cv2.setMouseCallback(window, mouse_callback)

    while True:
        frame = draw_view(image, boxes, selected, active_cls, index, len(images), image_path, scale)
        if dragging and drag_start is not None and drag_current is not None:
            x1, y1 = to_display_point(drag_start[0], drag_start[1], scale)
            x2, y2 = to_display_point(drag_current[0], drag_current[1], scale)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS.get(active_cls, (255, 255, 255)), 1)
        cv2.imshow(window, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            save_current()
            break
        if key == ord("s"):
            save_current()
        elif key in (ord("n"), ord("d"), 83):
            save_current()
            if index < len(images) - 1:
                index += 1
                load_current()
        elif key in (ord("p"), ord("a"), 81):
            save_current()
            if index > 0:
                index -= 1
                load_current()
        elif key in (ord("0"), ord("1"), ord("2")):
            active_cls = int(chr(key))
            if selected is not None and 0 <= selected < len(boxes):
                boxes[selected].cls = active_cls
                dirty = True
        elif key in (8, 127):
            if selected is not None and 0 <= selected < len(boxes):
                del boxes[selected]
                selected = None
                dirty = True
        elif key == ord("r"):
            load_current()

    if dirty:
        save_current()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
