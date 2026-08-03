"""Guarded preprocessing entry point for MIMIC-derived feature generation.

The original extraction implementation is not included in this submission.
This script documents the required inputs and protects the checked-in
`data/patient_features_ai.csv` file from accidental overwrite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from check_mimic_files import EXPECTED_FILES, check_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DATASET = PROJECT_ROOT / "data" / "patient_features_ai.csv"


def resolve_output(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Documented guard for rebuilding the processed ICU feature table. "
            "The original chunked extraction logic is not included."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing raw MIMIC-III CSV files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/patient_features_rebuilt.csv"),
        help="Output CSV path. Defaults to data/patient_features_rebuilt.csv.",
    )
    parser.add_argument("--allow-overwrite-runtime", action="store_true", help="Allow targeting data/patient_features_ai.csv.")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_path = resolve_output(args.output)

    if output_path == DEFAULT_RUNTIME_DATASET.resolve() and not args.allow_overwrite_runtime:
        print("Refusing to overwrite data/patient_features_ai.csv.")
        print("Choose a different --output path or explicitly pass --allow-overwrite-runtime.")
        return 2

    present, missing = check_files(source_dir)
    print(f"Expected raw files: {', '.join(EXPECTED_FILES)}")
    for line in present:
        print(f"OK  {line}")
    for filename in missing:
        print(f"MISS {filename}")
    if missing:
        print("Cannot proceed because required raw files are missing.")
        return 1

    print("Raw inputs appear to be present, but the original feature extraction implementation is not included.")
    print("No preprocessing was run and no output file was written.")
    print("Regenerating the dataset would require restoring the original chunked extraction logic, then retraining and retesting the model.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
