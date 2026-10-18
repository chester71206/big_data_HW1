"""Execute one assignment notebook and save its outputs under Assignment1_results."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="Notebook path to execute")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Assignment1_results"),
        help="Directory for executed notebook and experiment outputs",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Maximum time in seconds allowed for a single notebook cell",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    output_path = args.output_dir / f"{args.notebook.stem}_executed.ipynb"
    print(f"Executing {args.notebook}; output will be saved to {output_path}", flush=True)
    try:
        client.execute()
    finally:
        # Keep cell outputs from successful cells even if a later cell fails.
        nbformat.write(notebook, output_path)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
