from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.interaction.train_model import train_interaction_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backward-compatible shim for interaction training.")
    parser.add_argument("--data-dir", type=Path, default=Path("interaction_data"))
    parser.add_argument("--raw-data", type=Path, default=None)
    parser.add_argument("--encoded-data", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/interaction"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Run a smaller quick-mode training pass.")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: dict[str, object] = {
        "data_dir": args.data_dir,
        "model_dir": args.artifact_dir,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "quick": args.quick,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
    }
    if args.raw_data is not None:
        cfg["raw_data_path"] = args.raw_data
    if args.encoded_data is not None:
        cfg["encoded_data_path"] = args.encoded_data

    report = train_interaction_model(cfg)

    print(json.dumps(report, indent=2))
    print(f"\nSaved interaction artifacts to: {args.artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
