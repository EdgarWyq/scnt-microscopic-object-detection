from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from common import CLASS_NAMES, IMAGE_EXTENSIONS, ensure_dir, list_images, project_root, resolve_path, update_marked_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check SCNT YOLO dataset format.")
    parser.add_argument("--dataset-root", default="dataset/SCNT", help="Root directory containing SCNT-Source and SCNT-Target.")
    parser.add_argument("--output", default="outputs/dataset_check.txt", help="Text report output path.")
    parser.add_argument("--report", default="REPORT.md", help="Optional REPORT.md path to update.")
    parser.add_argument("--max-report-items", type=int, default=30, help="Maximum items per issue type inserted into REPORT.md.")
    return parser.parse_args()


def label_files(labels_dir: Path) -> List[Path]:
    if not labels_dir.exists():
        return []
    return sorted(p for p in labels_dir.rglob("*.txt") if p.name.lower() != "classes.txt")


def check_yolo_label(label_path: Path) -> Tuple[List[str], Counter]:
    issues: List[str] = []
    class_counter: Counter = Counter()
    text = label_path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        issues.append("empty_label")
        return issues, class_counter

    for line_no, line in enumerate(lines, start=1):
        parts = line.split()
        location = "{}:{}".format(label_path.as_posix(), line_no)
        if len(parts) != 5:
            issues.append("malformed_line | {} | {}".format(location, line))
            continue

        cls_text, *coord_text = parts
        try:
            cls = int(cls_text)
        except ValueError:
            issues.append("invalid_class | {} | {}".format(location, cls_text))
            continue

        if cls not in CLASS_NAMES:
            issues.append("invalid_class | {} | {}".format(location, cls))
        else:
            class_counter[cls] += 1

        try:
            coords = [float(v) for v in coord_text]
        except ValueError:
            issues.append("non_numeric_bbox | {} | {}".format(location, " ".join(coord_text)))
            continue

        x, y, w, h = coords
        if any(v < 0.0 or v > 1.0 for v in coords):
            issues.append("invalid_coord_range | {} | {}".format(location, " ".join(coord_text)))
        if w <= 0.0 or h <= 0.0:
            issues.append("non_positive_bbox | {} | {}".format(location, " ".join(coord_text)))

    return issues, class_counter


def check_split(split_name: str, split_dir: Path) -> Dict[str, object]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images = list_images(images_dir)
    labels = label_files(labels_dir)

    image_stems = {p.relative_to(images_dir).with_suffix("").as_posix(): p for p in images}
    label_stems = {p.relative_to(labels_dir).with_suffix("").as_posix(): p for p in labels}

    missing_labels = sorted(set(image_stems) - set(label_stems))
    extra_labels = sorted(set(label_stems) - set(image_stems))

    issue_map: Dict[str, List[str]] = defaultdict(list)
    class_counter: Counter = Counter()
    for stem in sorted(set(image_stems) & set(label_stems)):
        issues, counts = check_yolo_label(label_stems[stem])
        class_counter.update(counts)
        for issue in issues:
            issue_key = issue.split("|", 1)[0].strip()
            issue_map[issue_key].append(issue)

    for stem in missing_labels:
        issue_map["missing_label"].append(image_stems[stem].as_posix())
    for stem in extra_labels:
        issue_map["extra_label"].append(label_stems[stem].as_posix())

    duplicate_image_stems = [
        stem for stem, count in Counter(p.stem for p in images).items() if count > 1
    ]
    if duplicate_image_stems:
        issue_map["duplicate_image_stem"].extend(duplicate_image_stems)

    return {
        "split_name": split_name,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "image_count": len(images),
        "label_count": len(labels),
        "class_counter": class_counter,
        "issues": dict(issue_map),
    }


def render_text_report(results: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("SCNT dataset check")
    lines.append("=" * 80)
    lines.append("")
    for result in results:
        lines.append("[{}]".format(result["split_name"]))
        lines.append("images_dir: {}".format(result["images_dir"]))
        lines.append("labels_dir: {}".format(result["labels_dir"]))
        lines.append("images: {}".format(result["image_count"]))
        lines.append("labels: {}".format(result["label_count"]))
        lines.append("class counts:")
        counts: Counter = result["class_counter"]
        for cls, name in CLASS_NAMES.items():
            lines.append("  {} {}: {}".format(cls, name, counts.get(cls, 0)))
        issues: Dict[str, List[str]] = result["issues"]
        if not issues:
            lines.append("issues: none")
        else:
            lines.append("issues:")
            for issue_type in sorted(issues):
                items = issues[issue_type]
                lines.append("  {}: {}".format(issue_type, len(items)))
                for item in items:
                    lines.append("    - {}".format(item))
        lines.append("")
    return "\n".join(lines)


def render_report_section(results: List[Dict[str, object]], max_items: int) -> str:
    lines: List[str] = []
    any_issue = any(result["issues"] for result in results)
    if not any_issue:
        lines.append("数据检查已完成：未发现缺失标签、空标签、非法类别或非法坐标。详细结果见 `outputs/dataset_check.txt`。")
        return "\n".join(lines)

    lines.append("数据检查发现以下问题，完整清单见 `outputs/dataset_check.txt`：")
    for result in results:
        issues: Dict[str, List[str]] = result["issues"]
        if not issues:
            continue
        lines.append("")
        lines.append("- `{}`：".format(result["split_name"]))
        for issue_type in sorted(issues):
            items = issues[issue_type]
            preview = items[:max_items]
            lines.append("  - `{}`：{} 项".format(issue_type, len(items)))
            for item in preview:
                lines.append("    - `{}`".format(item))
            if len(items) > max_items:
                lines.append("    - ... 其余 {} 项见完整检查文件".format(len(items) - max_items))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_root = resolve_path(args.dataset_root, root)
    output_path = resolve_path(args.output, root)

    results = []
    for split_name in ["SCNT-Source", "SCNT-Target"]:
        split_dir = dataset_root / split_name
        results.append(check_split(split_name, split_dir))

    report_text = render_text_report(results)
    ensure_dir(output_path.parent)
    output_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print("Saved dataset check to {}".format(output_path))

    if args.report:
        report_path = resolve_path(args.report, root)
        section = render_report_section(results, args.max_report_items)
        update_marked_section(
            report_path,
            "<!-- DATASET_CHECK_START -->",
            "<!-- DATASET_CHECK_END -->",
            section,
        )
        print("Updated dataset-check section in {}".format(report_path))


if __name__ == "__main__":
    main()

