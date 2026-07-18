from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (
    "outputs",
    "multirun",
    "ray_results",
    ".pytest_tmp",
    ".pytest_cache",
    ".cache",
    "wandb_downloads",
)
PROTECTED_PREFIXES = (
    ROOT / "experiments",
    ROOT / "experiments" / "archive",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely remove known local artifact directories from the repo root.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting anything.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion of the known local artifact directories.",
    )
    return parser.parse_args()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_target(path: Path) -> str | None:
    resolved = path.resolve()
    if not resolved.exists():
        return "missing"
    if not resolved.is_dir():
        return "not a directory"
    if not _is_relative_to(resolved, ROOT):
        return "outside repo root"
    for protected in PROTECTED_PREFIXES:
        if resolved == protected.resolve() or _is_relative_to(resolved, protected.resolve()):
            return "protected source directory"
    return None


def _print_action(status: str, path: Path, detail: str = "") -> None:
    suffix = f" ({detail})" if detail else ""
    print(f"{status}: {path.relative_to(ROOT).as_posix()}{suffix}")


def main() -> int:
    args = _parse_args()
    targets = [ROOT / name for name in DEFAULT_TARGETS]

    if not args.dry_run and not args.yes:
        print("Refusing to delete artifacts without --yes. Use --dry-run to preview changes.")
        return 2

    for target in targets:
        problem = _validate_target(target)
        if problem is not None:
            _print_action("skipped", target, problem)
            continue
        if args.dry_run:
            _print_action("would remove", target)
            continue
        shutil.rmtree(target)
        _print_action("removed", target)

    return 0


if __name__ == "__main__":
    sys.exit(main())
