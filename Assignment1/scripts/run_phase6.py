from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "Assignment1" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from advml_assignment1 import PhaseSixConfig, PhaseSixPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Assignment 1 Phase 6 rolling backtesting.")
    parser.add_argument(
        "--phase3-design",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase3" / "phase3_design_data.csv",
        help="Path to the Phase 3 design matrix.",
    )
    parser.add_argument(
        "--phase5-design",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase5" / "phase5_design_data.csv",
        help="Path to the Phase 5 design matrix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase6",
        help="Directory where Phase 6 outputs will be written.",
    )
    parser.add_argument("--train-window", type=int, default=2000, help="Number of observations in each rolling training window.")
    parser.add_argument("--validation-window", type=int, default=252, help="Number of validation observations per fold.")
    parser.add_argument("--test-window", type=int, default=252, help="Number of test observations per fold.")
    parser.add_argument("--step-size", type=int, default=252, help="How far the rolling window advances after each fold.")
    parser.add_argument("--commission-bps", type=float, default=2.0, help="Commission cost in basis points per unit turnover.")
    parser.add_argument("--slippage-bps", type=float, default=3.0, help="Slippage cost in basis points per unit turnover.")
    parser.add_argument("--phase5-max-epochs", type=int, default=10, help="Maximum epochs for each Phase 5 fold retraining.")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the backtest without writing outputs to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhaseSixConfig(
        phase3_design_path=args.phase3_design,
        phase5_design_path=args.phase5_design,
        output_dir=args.output_dir,
        train_window=args.train_window,
        validation_window=args.validation_window,
        test_window=args.test_window,
        step_size=args.step_size,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        phase5_max_epochs=args.phase5_max_epochs,
    )
    artifacts = PhaseSixPipeline(config).run(save_outputs=not args.no_save)

    print(artifacts.summary())
    print("\nFold metrics:")
    print(artifacts.fold_metrics.to_string(index=False))
    print("\nStrategy summary:")
    print(artifacts.strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
