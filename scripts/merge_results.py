"""Consolidate several results CSV files into one, deduplicated by run name.

Each experiment run appends a row to a results CSV. When runs are executed
across several machines or accounts, their CSV files must be merged into a
single file for the dashboard and analysis. Rows are deduplicated by the unique
run name; earlier inputs take precedence on conflict, and any run whose recorded
values differ between inputs is reported so it can be re-checked.

Usage:
    python scripts/merge_results.py --inputs runs/results.csv runs/results-1.csv \
        --output runs/results.csv
"""

import argparse

import pandas as pd

_RUN_COLUMN = "run"
_SOURCE_ORDER = "_source_order"


def merge_results(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Merge results frames in precedence order (earlier wins on conflict).

    Returns the deduplicated frame sorted by run name, plus the sorted list of
    run names that appeared in more than one input with differing values.
    """
    if not frames:
        raise ValueError("merge_results requires at least one frame.")

    tagged = []
    for order, frame in enumerate(frames):
        if _RUN_COLUMN not in frame.columns:
            raise ValueError(f"Results frame {order} is missing a '{_RUN_COLUMN}' column.")
        annotated = frame.copy()
        annotated[_SOURCE_ORDER] = order
        tagged.append(annotated)

    combined = pd.concat(tagged, ignore_index=True)
    value_columns = [column for column in combined.columns if column != _SOURCE_ORDER]

    conflicts = []
    for run_name, group in combined.groupby(_RUN_COLUMN):
        if group[_SOURCE_ORDER].nunique() <= 1:
            continue
        if len(group[value_columns].drop_duplicates()) > 1:
            conflicts.append(str(run_name))

    merged = (
        combined.drop_duplicates(subset=_RUN_COLUMN, keep="first")
        .drop(columns=_SOURCE_ORDER)
        .sort_values(_RUN_COLUMN)
        .reset_index(drop=True)
    )
    return merged, sorted(conflicts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge results CSV files into one.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input results CSV paths, in precedence order (earlier wins on conflict).",
    )
    parser.add_argument("--output", required=True, help="Path for the merged results CSV.")
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.inputs]
    merged, conflicts = merge_results(frames)
    merged.to_csv(args.output, index=False)

    print(f"Merged {len(args.inputs)} file(s) into {args.output}: {len(merged)} unique runs.")
    if conflicts:
        print(
            f"{len(conflicts)} run(s) differed between inputs "
            f"(kept the earlier input's row): {', '.join(conflicts)}"
        )


if __name__ == "__main__":
    main()
