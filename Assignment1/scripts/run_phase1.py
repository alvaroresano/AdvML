from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "Assignment1" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from advml_assignment1 import PhaseOneConfig, PhaseOnePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Assignment 1 phase-one preprocessing.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "financial_regression.csv",
        help="Path to the raw Kaggle CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase1",
        help="Directory where derived datasets and ADF summaries will be written.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the pipeline without writing CSV artifacts to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhaseOneConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )
    artifacts = PhaseOnePipeline(config).run(save_outputs=not args.no_save)

    print(artifacts.summary())
    print("\nADF summary:")
    print(
        artifacts.adf_summary[
            ["asset", "transformation", "adf_statistic", "p_value", "reject_unit_root_5pct", "status"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
