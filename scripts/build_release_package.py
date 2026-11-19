from __future__ import annotations

import argparse
import json
from pathlib import Path

from sciprism_x.release.package import (
    audit_formal_release_run,
    build_release_package,
    load_release_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the allowlisted SciPRISM-X release package"
    )
    parser.add_argument("--run", type=Path, required=True, help="completed formal run root")
    parser.add_argument("--target", type=Path, required=True, help="new package directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "release_package.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt_path = args.run / "formal_run_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("formal run receipt is required before package construction")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    scale = str(receipt.get("scale", ""))
    if scale not in {"smoke", "paper"}:
        raise ValueError("formal run receipt has an invalid scale")
    audit = audit_formal_release_run(args.run, scale=scale)
    if audit.errors:
        raise ValueError(f"formal release contract audit failed: {audit.errors}")
    target = build_release_package(
        project_root=PROJECT_ROOT,
        run_root=args.run,
        target=args.target,
        config=load_release_config(args.config),
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
