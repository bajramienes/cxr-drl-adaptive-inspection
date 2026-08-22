from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, List

import pandas as pd

PROJECT_DIR = Path(r"C:\Users\Ensar\Desktop\CXR8\cxr_drl")
DEFAULT_RESULTS = PROJECT_DIR / "outputs_fast_48h"
OUTPUT_NAME = "summary_final_results.txt"

TABLE_EXTS = {".csv", ".tsv"}
EXCEL_EXTS = {".xlsx", ".xls", ".xlsm"}
TEXT_EXTS = {".txt", ".log", ".md", ".yaml", ".yml", ".ini", ".cfg"}
CHECKPOINT_EXTS = {".pt", ".pth", ".ckpt"}


def human_size(n: int) -> str:
    value = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{n} B"


def clean(value: Any, limit: int = 180) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def line(char: str = "=", width: int = 100) -> str:
    return char * width


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def count_table_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return -1


def inspect_csv(path: Path) -> List[str]:
    out: List[str] = []
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        frame = pd.read_csv(path, sep=sep, nrows=2, low_memory=False)
        rows = count_table_rows(path)
        out.append(f"Rows excluding header: {rows:,}" if rows >= 0 else "Rows: unavailable")
        out.append(f"Column count: {len(frame.columns)}")
        out.append("Headers:")
        for i, col in enumerate(frame.columns, 1):
            out.append(f"  {i:02d}. {col}")
        out.append("First two rows:")
        if frame.empty:
            out.append("  <none>")
        else:
            for idx, row in frame.iterrows():
                values = " | ".join(f"{c}={clean(row[c])}" for c in frame.columns)
                out.append(f"  Row {idx + 1}: {values}")
    except Exception as exc:
        out.append(f"WARNING: Could not read table: {type(exc).__name__}: {exc}")
    return out


def inspect_excel(path: Path) -> List[str]:
    out: List[str] = []
    try:
        book = pd.ExcelFile(path)
        out.append(f"Sheets ({len(book.sheet_names)}): {', '.join(book.sheet_names)}")
        for sheet in book.sheet_names:
            try:
                frame = pd.read_excel(path, sheet_name=sheet, nrows=2)
                out.append(f"Sheet: {sheet}")
                out.append(f"  Column count: {len(frame.columns)}")
                out.append("  Headers:")
                for i, col in enumerate(frame.columns, 1):
                    out.append(f"    {i:02d}. {col}")
                if not frame.empty:
                    out.append("  First two rows:")
                    for idx, row in frame.iterrows():
                        values = " | ".join(f"{c}={clean(row[c])}" for c in frame.columns)
                        out.append(f"    Row {idx + 1}: {values}")
            except Exception as exc:
                out.append(f"  WARNING: Could not read sheet {sheet}: {exc}")
    except Exception as exc:
        out.append(f"WARNING: Could not read workbook: {type(exc).__name__}: {exc}")
    return out


def compact_json(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        if isinstance(value, dict):
            return f"<dict with {len(value)} keys>"
        if isinstance(value, list):
            return f"<list with {len(value)} items>"
        return value
    if isinstance(value, dict):
        result = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 25:
                result["..."] = f"{len(value) - 25} more keys"
                break
            result[str(k)] = compact_json(v, depth + 1)
        return result
    if isinstance(value, list):
        result = [compact_json(v, depth + 1) for v in value[:5]]
        if len(value) > 5:
            result.append(f"<{len(value) - 5} more items>")
        return result
    return value


def inspect_json(path: Path) -> List[str]:
    out: List[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            out.append("Top-level keys: " + ", ".join(map(str, data.keys())))
        elif isinstance(data, list):
            out.append(f"Top-level list length: {len(data)}")
        preview = json.dumps(compact_json(data), indent=2, ensure_ascii=False)
        if len(preview) > 2500:
            preview = preview[:2500] + "\n...<truncated>"
        out.append("Preview:")
        out.extend("  " + item for item in preview.splitlines())
    except Exception as exc:
        out.append(f"WARNING: Could not read JSON: {type(exc).__name__}: {exc}")
    return out


def inspect_text(path: Path) -> List[str]:
    out: List[str] = []
    try:
        preview = []
        total = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for text_line in f:
                total += 1
                if len(preview) < 12:
                    preview.append(text_line.rstrip())
        out.append(f"Text lines: {total:,}")
        out.append("First lines:")
        out.extend("  " + clean(item, 300) for item in preview)
        if not preview:
            out.append("  <empty>")
    except Exception as exc:
        out.append(f"WARNING: Could not read text: {type(exc).__name__}: {exc}")
    return out


def inspect_checkpoint(path: Path) -> List[str]:
    out: List[str] = []
    try:
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        out.append(f"Object type: {type(data).__name__}")
        if isinstance(data, dict):
            keys = list(data.keys())
            out.append(f"Top-level keys ({len(keys)}): {', '.join(map(str, keys[:50]))}")
            for key in keys[:50]:
                value = data[key]
                if hasattr(value, "shape"):
                    out.append(f"  {key}: shape={tuple(value.shape)}")
                elif isinstance(value, dict):
                    out.append(f"  {key}: dict with {len(value)} keys")
                elif isinstance(value, (list, tuple)):
                    out.append(f"  {key}: {type(value).__name__} with {len(value)} items")
                elif isinstance(value, (str, int, float, bool, type(None))):
                    out.append(f"  {key}: {clean(value)}")
                else:
                    out.append(f"  {key}: {type(value).__name__}")
    except Exception as exc:
        out.append(f"WARNING: Could not inspect checkpoint: {type(exc).__name__}: {exc}")
    return out


def inspect_file(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix in TABLE_EXTS:
        return inspect_csv(path)
    if suffix in EXCEL_EXTS:
        return inspect_excel(path)
    if suffix == ".json":
        return inspect_json(path)
    if suffix in TEXT_EXTS:
        return inspect_text(path)
    if suffix in CHECKPOINT_EXTS:
        return inspect_checkpoint(path)
    return ["No content preview configured for this file type."]


def completeness(results: Path) -> List[str]:
    out: List[str] = []
    expected_folders = ["checkpoints", "feature_cache", "figures", "logs", "state", "summaries"]
    for name in expected_folders:
        path = results / name
        out.append(f"[{'OK' if path.is_dir() else 'MISSING'}] Folder: {path}")

    summary_candidates = [
        results / "summaries" / "drl_phase_summary.csv",
        results / "summaries" / "final_agent_summary.csv",
        results / "summaries" / "baseline_test_summary.csv",
        results / "summaries" / "final_summary.csv",
    ]
    for path in summary_candidates:
        out.append(f"[{'OK' if path.is_file() else 'MISSING'}] Summary: {path}")
    return out


def matrix_check(results: Path) -> List[str]:
    path = results / "summaries" / "drl_phase_summary.csv"
    if not path.exists():
        return [f"Matrix check unavailable: {path} not found."]
    try:
        frame = pd.read_csv(path)
        needed = {"agent", "seed", "phase"}
        missing_cols = needed.difference(frame.columns)
        if missing_cols:
            return ["Matrix check unavailable; missing columns: " + ", ".join(sorted(missing_cols))]
        expected = {
            (agent, seed, phase)
            for agent in ["PPO", "DQN", "DiscreteSAC"]
            for seed in [5, 25, 125]
            for phase in ["early", "mid", "final"]
        }
        observed = {
            (str(row.agent), int(row.seed), str(row.phase))
            for row in frame.itertuples(index=False)
        }
        out = [
            f"Expected combinations: {len(expected)}",
            f"Observed unique combinations: {len(observed)}",
            f"Duplicate rows: {int(frame.duplicated(['agent', 'seed', 'phase']).sum())}",
        ]
        missing = sorted(expected - observed)
        if missing:
            out.append("Missing combinations:")
            out.extend(f"  agent={a}, seed={s}, phase={p}" for a, s, p in missing)
        else:
            out.append("All 27 agent/seed/phase combinations are present.")
        return out
    except Exception as exc:
        return [f"WARNING: Matrix check failed: {type(exc).__name__}: {exc}"]


def build_report(results: Path, output: Path) -> None:
    if not results.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results}")

    files = sorted(
        [p for p in results.rglob("*") if p.is_file() and p.resolve() != output.resolve()],
        key=lambda p: str(p).lower(),
    )

    ext_counts = Counter()
    ext_sizes = Counter()
    folder_counts = Counter()
    folder_sizes = Counter()
    total_size = 0
    empty_files = []

    for path in files:
        size = path.stat().st_size
        total_size += size
        ext = path.suffix.lower() or "<no extension>"
        ext_counts[ext] += 1
        ext_sizes[ext] += size
        parent = relative(path.parent, results)
        folder_counts[parent] += 1
        folder_sizes[parent] += size
        if size == 0:
            empty_files.append(path)

    report: List[str] = [
        line(),
        "FINAL EXPERIMENT RESULTS INVENTORY",
        line(),
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Project directory: {results.parent}",
        f"Results directory: {results}",
        f"Summary output: {output}",
        f"Total files: {len(files):,}",
        f"Total size: {human_size(total_size)}",
        "",
        line(),
        "EXPECTED OUTPUT COMPLETENESS",
        line(),
        *completeness(results),
        "",
        line(),
        "DRL EXPERIMENT MATRIX CHECK",
        line(),
        *matrix_check(results),
        "",
        line(),
        "FILE TYPES",
        line(),
    ]

    for ext in sorted(ext_counts):
        report.append(f"{ext:<15} files={ext_counts[ext]:>6,} | size={human_size(ext_sizes[ext])}")

    report.extend(["", line(), "FOLDER SUMMARY", line()])
    for folder in sorted(folder_counts):
        report.append(f"{folder}\n  Files: {folder_counts[folder]:,}\n  Size: {human_size(folder_sizes[folder])}")

    report.extend(["", line(), "EMPTY FILE CHECK", line()])
    if empty_files:
        report.append(f"Empty files: {len(empty_files)}")
        report.extend(f"  {p}" for p in empty_files)
    else:
        report.append("No empty files detected.")

    report.extend(["", line(), "COMPLETE FILE INVENTORY AND CONTENT HEADERS", line()])

    for i, path in enumerate(files, 1):
        stat = path.stat()
        report.extend([
            "",
            line("-"),
            f"FILE {i:,} OF {len(files):,}",
            line("-"),
            f"Name: {path.name}",
            f"Relative location: {relative(path, results)}",
            f"Absolute location: {path}",
            f"Extension: {path.suffix.lower() or '<none>'}",
            f"Size: {human_size(stat.st_size)} ({stat.st_size:,} bytes)",
            f"Modified: {datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds')}",
            "Content/header inspection:",
        ])
        report.extend("  " + item for item in inspect_file(path))

    report.extend(["", line(), "END OF REPORT", line()])
    output.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary_final_results.txt from all experiment files.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else results / OUTPUT_NAME

    print("=" * 72)
    print("FINAL RESULTS INVENTORY GENERATOR")
    print("=" * 72)
    print(f"Reading: {results}")
    print(f"Writing: {output}")

    build_report(results, output)

    print("\nCompleted successfully.")
    print(f"Summary created at:\n{output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
