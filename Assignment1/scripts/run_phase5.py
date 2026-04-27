from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "Assignment1" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from advml_assignment1 import PhaseFiveConfig, PhaseFivePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Assignment 1 Phase 5 deep forecasting benchmark.")
    parser.add_argument(
        "--input-data",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase1" / "modeling_data.csv",
        help="Path to the Phase 1 modeling dataset.",
    )
    parser.add_argument(
        "--phase3-metadata",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase3" / "model_metadata.json",
        help="Path to the Phase 3 metadata file used for benchmark comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Assignment1" / "outputs" / "phase5",
        help="Directory where Phase 5 outputs will be written.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=40,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size used during training.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the pipeline without writing outputs to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhaseFiveConfig(
        input_data_path=args.input_data,
        phase3_metadata_path=args.phase3_metadata,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )
    artifacts = PhaseFivePipeline(config).run(save_outputs=not args.no_save)

    print(artifacts.summary())
    print("\nLatest training history:")
    print(artifacts.training_history.tail(10).to_string(index=False))
    print("\nTest forecast sample:")
    print(artifacts.test_predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
