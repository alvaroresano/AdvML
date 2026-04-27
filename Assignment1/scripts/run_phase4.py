from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "Assignment1" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from advml_assignment1 import PhaseFourConfig, PhaseFourPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Assignment 1 Phase 4 GARCH volatility modeling.")
    parser.add_argument(
        "--phase3-train",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase3" / "train_fitted.csv",
        help="Path to the Phase 3 training residual file.",
    )
    parser.add_argument(
        "--phase3-test",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase3" / "test_forecasts.csv",
        help="Path to the Phase 3 out-of-sample forecast file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase4",
        help="Directory where Phase 4 outputs will be written.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the pipeline without writing outputs to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhaseFourConfig(
        phase3_train_path=args.phase3_train,
        phase3_test_path=args.phase3_test,
        output_dir=args.output_dir,
    )
    artifacts = PhaseFourPipeline(config).run(save_outputs=not args.no_save)

    print(artifacts.summary())
    print("\nKey GARCH parameters:")
    print(artifacts.parameter_summary.to_string(index=False))
    print("\nResidual diagnostics:")
    print(artifacts.residual_diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
