from __future__ import annotations

import argparse
from pathlib import Path

from sciprism_x.release.package import audit_formal_release_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the Deliverables 1-8 formal release artifact contract"
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("--scale", choices=("smoke", "paper"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_formal_release_run(args.run, scale=args.scale)
    print(f"checked_artifacts={audit.checked_artifacts}")
    print(f"checked_instances={audit.checked_instances}")
    print(f"errors={len(audit.errors)}")
    for error in audit.errors:
        print(error)
    return 1 if audit.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
