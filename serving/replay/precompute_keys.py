"""Materialize CPU draft keys before benchmark runs."""

import argparse
import json
from pathlib import Path

from selector_workers import precompute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    data = json.loads(args.workload.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    stats = precompute(data, args.output, args.workers)
    (args.output / "precompute-stats.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
