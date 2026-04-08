from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorSpecification:
    output_files: tuple[str, ...] = (
        "dataset_v2.csv",
        "dataset_encoded_v2.csv",
        "dataset_v2.json",
        "dataset_v2_quality_report.json",
    )
    total_sessions: int = 10000
    version: str = "v2"


SPEC = GeneratorSpecification()


def build_generator_specification() -> dict:
    return {
        "generator_version": SPEC.version,
        "total_sessions": SPEC.total_sessions,
        "output_files": list(SPEC.output_files),
        "status": "inactive",
        "message": "External datasets must be provided by the user. Synthetic generation is intentionally disabled.",
    }


def generate_dataset(*args, **kwargs):
    raise RuntimeError(
        "synthetic_dataset_generator_v2.py is intentionally inactive. "
        "Provide an external CSV dataset and run the training pipeline against it."
    )


def main():
    raise SystemExit(
        "Synthetic data generation is disabled for this project. Supply an external CSV dataset instead."
    )


if __name__ == "__main__":
    main()
