"""Check whether a local MIMIC-III directory has the expected raw CSV files.

This script only inspects file names and sizes. It does not copy raw MIMIC
tables, read large tables, or write processed outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_FILES = [
    "PATIENTS.csv",
    "ADMISSIONS.csv",
    "ICUSTAYS.csv",
    "D_ITEMS.csv",
    "D_LABITEMS.csv",
    "CHARTEVENTS.csv",
    "LABEVENTS.csv",
]


def format_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} TB"


def check_files(source_dir: Path) -> tuple[list[str], list[str]]:
    present = []
    missing = []
    for filename in EXPECTED_FILES:
        path = source_dir / filename
        if path.exists():
            present.append(f"{filename}: found ({format_size(path)})")
        else:
            missing.append(filename)
    return present, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Check expected MIMIC-III CSV files.")
    parser.add_argument("source_dir", type=Path, help="Directory containing raw MIMIC-III CSV files.")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}")
        return 2

    present, missing = check_files(source_dir)
    print(f"Checked: {source_dir}")
    for line in present:
        print(f"OK  {line}")
    for filename in missing:
        print(f"MISS {filename}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
