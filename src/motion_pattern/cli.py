from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the motion clustering and anomaly detection demo pipeline."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/demo"))
    parser.add_argument("--samples", type=int, default=360)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_demo(output_dir=args.output_dir, n_samples=args.samples)
    print(json.dumps(result["anomaly_detection"], indent=2))


if __name__ == "__main__":
    main()
