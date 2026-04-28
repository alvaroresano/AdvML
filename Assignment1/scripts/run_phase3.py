from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "Assignment1" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from advml_assignment1 import PhaseThreeConfig, PhaseThreePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Assignment 1 Phase 3 classical SARIMAX baseline.")
    parser.add_argument(
        "--input-data",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase1" / "modeling_data.csv",
        help="Path to the Phase 1 complete-case modeling dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase3",
        help="Directory where Phase 3 outputs will be written.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="nasdaq log_return",
        help="Target return series to model.",
    )
    parser.add_argument(
        "--holdout-size",
        type=int,
        default=252,
        help="Number of final trading observations reserved for out-of-sample evaluation.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the pipeline without writing outputs to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhaseThreeConfig(
        input_data_path=args.input_data,
        output_dir=args.output_dir,
        target_column=args.target_column,
        holdout_size=args.holdout_size,
    )
    artifacts = PhaseThreePipeline(config).run(save_outputs=not args.no_save)

    print(artifacts.summary())
    print("\nResidual diagnostics:")
    print(artifacts.residual_diagnostics.to_string(index=False))
    print("\nTop coefficients by absolute magnitude:")
    display_frame = artifacts.coefficient_summary.copy()
    display_frame["abs_coefficient"] = display_frame["coefficient"].abs()
    print(
        display_frame.sort_values("abs_coefficient", ascending=False)[
            ["parameter", "coefficient", "p_value", "ci_lower", "ci_upper"]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
