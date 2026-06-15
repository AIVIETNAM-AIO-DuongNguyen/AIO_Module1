"""Generate run configs from a matrix spec and a Kaggle-account assignment.

Usage:

    python scripts/generate_matrix.py --matrix configs/matrix/core.yaml \
        --out-dir configs/core --accounts 5
"""

import argparse
import csv
import os
import sys

import yaml

# Allow running as a standalone script by putting the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.matrix import assign_slices, generate_configs, write_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate run configs from a matrix spec.")
    parser.add_argument("--matrix", required=True, help="Path to the matrix spec YAML.")
    parser.add_argument("--out-dir", default="configs/core", help="Directory for generated configs.")
    parser.add_argument("--accounts", type=int, default=5, help="Number of compute accounts to split runs across.")
    parser.add_argument("--manifest", default=None, help="Path for the assignment CSV (defaults to <out-dir>/assignment.csv).")
    args = parser.parse_args()

    with open(args.matrix, "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    configs = generate_configs(spec)
    paths = write_configs(configs, args.out_dir)
    assignments = assign_slices([config["run"]["name"] for config in configs], args.accounts)

    manifest = args.manifest or os.path.join(args.out_dir, "assignment.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "config_path", "account"])
        writer.writeheader()
        for assignment, path in zip(assignments, paths):
            writer.writerow(
                {"run": assignment["run"], "config_path": path, "account": assignment["account"]}
            )

    print(f"Generated {len(configs)} configs in {args.out_dir}")
    print(f"Assignment manifest: {manifest}")


if __name__ == "__main__":
    main()
