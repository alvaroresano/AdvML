from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "Assignment1" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from advml_assignment1 import PhaseTwoConfig, PhaseTwoPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Assignment 1 Phase 2 STL decomposition.")
    parser.add_argument(
        "--input-data",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase1" / "cleaned_data.csv",
        help="Path to the cleaned trading-calendar dataset from Phase 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase2",
        help="Directory where STL summaries, components, and plots will be written.",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=5,
        help="Seasonal period measured in trading observations. Use 5 for a trading week.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the pipeline without writing CSV artifacts to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhaseTwoConfig(
        input_data_path=args.input_data,
        output_dir=args.output_dir,
        stl_period=args.period,
    )
    artifacts = PhaseTwoPipeline(config).run(save_outputs=not args.no_save)

    print(artifacts.summary())
    print("\nSTL summary:")
    print(
        artifacts.decomposition_summary[
            [
                "asset",
                "trend_strength",
                "seasonal_strength",
                "seasonal_amplitude",
                "residual_std",
                "residual_share_of_variance",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
