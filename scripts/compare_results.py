from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from common import CLASS_NAMES, latest_rows_by_experiment, project_root, read_csv_rows, resolve_path, update_marked_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and pseudo-adaptation metrics and update REPORT.md.")
    parser.add_argument("--summary-csv", default="outputs/experiments_summary.csv")
    parser.add_argument("--report", default="REPORT.md")
    parser.add_argument("--artifact-root", default="runs/scnt")
    return parser.parse_args()


def fmt(value: str) -> str:
    if value in (None, ""):
        return "-"
    try:
        return "{:.4f}".format(float(value))
    except ValueError:
        return value


def diff(new_value: str, old_value: str) -> str:
    try:
        return "{:+.4f}".format(float(new_value) - float(old_value))
    except (TypeError, ValueError):
        return "-"


def latest_matching(rows_by_exp: Dict[str, Dict[str, str]], prefixes: List[str]) -> Dict[str, str]:
    for prefix in prefixes:
        matches = [
            row for exp, row in rows_by_exp.items()
            if exp == prefix or exp.startswith(prefix + "_") or exp.startswith(prefix + "-")
        ]
        if matches:
            return matches[-1]
    return {}


def markdown_table(rows_by_exp: Dict[str, Dict[str, str]]) -> str:
    display_rows = [
        ("source_plain", latest_matching(rows_by_exp, ["source_plain", "baseline"])),
        ("source_aug", latest_matching(rows_by_exp, ["source_aug"])),
        ("pseudo_adapt", latest_matching(rows_by_exp, ["pseudo_adapt"])),
    ]
    header = [
        "Experiment",
        "Run",
        "AP50 injection",
        "AP50 holding",
        "AP50 oocyte",
        "mAP50",
        "mAP50-95",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for display_name, row in display_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                display_name,
                row.get("experiment", "-") if row else "-",
                fmt(row.get("ap50_injection_needle", "")),
                fmt(row.get("ap50_holding_needle", "")),
                fmt(row.get("ap50_oocyte", "")),
                fmt(row.get("map50", "")),
                fmt(row.get("map50_95", "")),
            )
        )

    base = latest_matching(rows_by_exp, ["source_aug", "source_plain", "baseline"])
    pseudo = latest_matching(rows_by_exp, ["pseudo_adapt"])
    if base and pseudo:
        lines.append(
            "| pseudo_adapt - teacher/baseline | - | {} | {} | {} | {} | {} |".format(
                diff(pseudo.get("ap50_injection_needle", ""), base.get("ap50_injection_needle", "")),
                diff(pseudo.get("ap50_holding_needle", ""), base.get("ap50_holding_needle", "")),
                diff(pseudo.get("ap50_oocyte", ""), base.get("ap50_oocyte", "")),
                diff(pseudo.get("map50", ""), base.get("map50", "")),
                diff(pseudo.get("map50_95", ""), base.get("map50_95", "")),
            )
        )
    return "\n".join(lines)


def artifact_section(root: Path) -> str:
    patterns = [
        "confusion_matrix*.png",
        "PR_curve*.png",
        "F1_curve*.png",
        "P_curve*.png",
        "R_curve*.png",
        "results.png",
    ]
    paths: List[Path] = []
    if root.exists():
        for pattern in patterns:
            paths.extend(sorted(root.rglob(pattern)))
    vis_root = project_root() / "outputs" / "visualizations"
    if vis_root.exists():
        paths.extend(sorted(vis_root.rglob("*.jpg"))[:10])
        paths.extend(sorted(vis_root.rglob("*.png"))[:10])

    if not paths:
        return "当前还没有检测到混淆矩阵、PR/F1 曲线或预测可视化图。训练和验证后重新运行 `compare_results.py` 会自动更新这里。"

    lines = ["可查看的结果文件："]
    for path in paths[:40]:
        try:
            rel = path.resolve().relative_to(project_root().resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        lines.append("- `{}`".format(rel))
    if len(paths) > 40:
        lines.append("- ... 其余结果文件位于 `{}`".format(root.as_posix()))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = project_root()
    summary_csv = resolve_path(args.summary_csv, root)
    report = resolve_path(args.report, root)
    artifact_root = resolve_path(args.artifact_root, root)

    rows = read_csv_rows(summary_csv)
    rows_by_exp = latest_rows_by_experiment(rows)
    table = markdown_table(rows_by_exp)
    artifacts = artifact_section(artifact_root)

    update_marked_section(report, "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->", table)
    update_marked_section(report, "<!-- ARTIFACTS_START -->", "<!-- ARTIFACTS_END -->", artifacts)

    print(table)
    print("Updated {}".format(report))


if __name__ == "__main__":
    main()
