"""Utilities for exporting W&B runs into reproducible local artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JSON_SEQUENCE_TYPES = list | tuple
JSON_SCALAR_TYPES = str | int | float | bool


def repo_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parent.parent


def resolve_env_file(explicit_path: str = "") -> Path | None:
    """Resolve the `.env` file path used to load W&B credentials."""

    candidates: list[Path] = []
    if explicit_path:
        path = Path(explicit_path).expanduser()
        candidates.append(path if path.is_absolute() else Path.cwd() / path)
        candidates.append(repo_root() / path.name)
    candidates.append(Path.cwd() / ".env")
    candidates.append(repo_root() / ".env")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_env_file(env_path: Path | None) -> None:
    """Load environment variables from an env file when one is available."""

    if env_path is None:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return
    load_dotenv(env_path, override=False)


def load_wandb_credentials(*, explicit_env_file: str = "") -> dict[str, str | None]:
    """Load W&B credentials and related defaults from the environment."""

    env_path = resolve_env_file(explicit_env_file)
    load_env_file(env_path)
    return {
        "env_path": str(env_path) if env_path is not None else None,
        "api_key": os.environ.get("WANDB_API_KEY"),
        "entity": os.environ.get("WANDB_ENTITY"),
        "project": os.environ.get("WANDB_PROJECT"),
    }


def require_all_tags(run_tags: Iterable[str], required_tags: Iterable[str]) -> bool:
    """Return whether a run includes every requested tag."""

    run_tag_set = {str(tag) for tag in run_tags}
    return all(str(tag) in run_tag_set for tag in required_tags)


def require_run_name(run_name: Any, required_names: Iterable[str]) -> bool:
    """Return whether a run name matches the requested allowlist."""

    required_name_set = {str(name) for name in required_names}
    if not required_name_set:
        return True
    return str(run_name or "") in required_name_set


def safe_slug(value: Any, *, fallback: str) -> str:
    """Convert an arbitrary value into a filesystem-safe slug."""

    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("._-")
    return text or fallback


def json_ready(value: Any) -> Any:
    """Convert nested values into JSON-serializable data."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, JSON_SEQUENCE_TYPES):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat() + "Z"
        return value.isoformat()
    if isinstance(value, JSON_SCALAR_TYPES) or value is None:
        return value
    try:
        item_method = getattr(value, "item")
    except Exception:
        item_method = None
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass
    try:
        tolist_method = getattr(value, "tolist")
    except Exception:
        tolist_method = None
    if callable(tolist_method):
        try:
            return tolist_method()
        except Exception:
            pass
    return str(value)


def extract_run_metadata(run: Any) -> dict[str, Any]:
    """Extract stable, serializable metadata from a W&B run object."""

    run_state = getattr(run, "state", None)
    run_tags = list(getattr(run, "tags", []) or [])
    created_at = getattr(run, "created_at", None)
    updated_at = getattr(run, "updated_at", None)
    project = getattr(getattr(run, "project", None), "name", None)
    entity = getattr(getattr(run, "entity", None), "name", None)

    metadata = {
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "path": list(getattr(run, "path", []) or []),
        "url": getattr(run, "url", None),
        "state": run_state,
        "tags": run_tags,
        "group": getattr(run, "group", None),
        "job_type": getattr(run, "job_type", None),
        "created_at": json_ready(created_at),
        "updated_at": json_ready(updated_at),
        "entity": entity,
        "project": project,
    }
    try:
        metadata["notes"] = getattr(run, "notes", None)
    except Exception:
        metadata["notes"] = None
    return json_ready(metadata)


def run_export_dir(output_dir: Path, entity: str, project: str, run: Any) -> Path:
    """Return the export directory for a specific run."""

    run_id = safe_slug(getattr(run, "id", None), fallback="run")
    run_name = safe_slug(getattr(run, "name", None), fallback="unnamed")
    return output_dir / safe_slug(entity, fallback="entity") / safe_slug(project, fallback="project") / f"{run_id}__{run_name}"


def iter_matching_runs(
    api: Any,
    entity: str,
    project: str,
    required_tags: list[str],
    required_names: list[str] | None = None,
) -> list[Any]:
    """Return runs whose tags and names satisfy the requested filters."""

    runs = api.runs(f"{entity}/{project}")
    required_names = list(required_names or [])
    return [
        run
        for run in runs
        if require_all_tags(getattr(run, "tags", []) or [], required_tags)
        and require_run_name(getattr(run, "name", None), required_names)
    ]


def export_run(run: Any, export_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Export one W&B run's metadata, config, summary, and history."""

    if export_dir.exists():
        if not overwrite:
            return {"status": "skipped", "path": str(export_dir)}
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    metadata = extract_run_metadata(run)
    config = json_ready(dict(getattr(run, "config", {}) or {}))
    summary = json_ready(dict(getattr(run, "summary", {}) or {}))

    history_path = export_dir / "history.jsonl"
    history_rows = 0
    with history_path.open("w", encoding="utf-8") as handle:
        for row in run.scan_history():
            handle.write(json.dumps(json_ready(dict(row)), sort_keys=True) + "\n")
            history_rows += 1

    (export_dir / "run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {"status": "exported", "path": str(export_dir), "history_rows": history_rows}


def download_runs(
    *,
    api: Any,
    entity: str,
    project: str,
    required_tags: list[str],
    required_names: list[str] | None,
    output_dir: Path,
    limit: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download matching runs and write a manifest for the export batch."""

    matching_runs = iter_matching_runs(api, entity, project, required_tags, required_names)
    if limit is not None:
        matching_runs = matching_runs[:limit]

    project_dir = output_dir / safe_slug(entity, fallback="entity") / safe_slug(project, fallback="project")
    project_dir.mkdir(parents=True, exist_ok=True)

    run_entries: list[dict[str, Any]] = []
    exported_count = 0
    skipped_count = 0

    for run in matching_runs:
        metadata = extract_run_metadata(run)
        export_dir = run_export_dir(output_dir, entity, project, run)
        entry = {
            "run_id": metadata["id"],
            "run_name": metadata["name"],
            "tags": metadata["tags"],
            "url": metadata["url"],
            "path": str(export_dir),
        }
        if dry_run:
            entry["status"] = "matched"
        else:
            result = export_run(run, export_dir, overwrite=overwrite)
            entry.update(result)
            if result["status"] == "exported":
                exported_count += 1
            else:
                skipped_count += 1
        run_entries.append(entry)

    manifest = {
        "entity": entity,
        "project": project,
        "requested_tags": list(required_tags),
        "requested_run_names": list(required_names or []),
        "dry_run": dry_run,
        "overwrite": overwrite,
        "limit": limit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matched_runs": len(matching_runs),
        "exported_runs": exported_count,
        "skipped_runs": skipped_count,
        "runs": run_entries,
    }
    (project_dir / "download_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the W&B download utility."""

    parser = argparse.ArgumentParser(description="Download W&B runs into local JSONL/JSON files for inspection.")
    parser.add_argument("--entity", default=None, help="W&B entity name. Defaults to WANDB_ENTITY from .env or env vars.")
    parser.add_argument("--project", default=None, help="W&B project name. Defaults to WANDB_PROJECT from .env or env vars.")
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=[],
        help="Required tag. Repeat the flag to require multiple tags.",
    )
    parser.add_argument(
        "--run-name",
        dest="run_names",
        action="append",
        default=[],
        help="Exact run name to keep. Repeat the flag to allow multiple names.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "wandb_downloads"),
        help="Destination root for downloaded runs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of matched runs processed.")
    parser.add_argument("--dry-run", action="store_true", help="Preview matching runs without downloading run payloads.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing exported run directory.")
    parser.add_argument(
        "--env-file",
        default="",
        help="Optional explicit .env path. Falls back to the repo-root .env lookup.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint for W&B run export."""

    args = build_parser().parse_args(argv)
    credentials = load_wandb_credentials(explicit_env_file=args.env_file)
    api_key = credentials["api_key"]
    entity = args.entity or credentials["entity"]
    project = args.project or credentials["project"]

    if not api_key:
        raise SystemExit("WANDB_API_KEY was not found in the repo .env or environment variables.")
    if not entity:
        raise SystemExit("W&B entity was not provided. Set WANDB_ENTITY or pass --entity.")
    if not project:
        raise SystemExit("W&B project was not provided. Set WANDB_PROJECT or pass --project.")

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("The `wandb` package is required to download runs. Install the project dependencies first.") from exc

    api = wandb.Api(api_key=api_key)
    manifest = download_runs(
        api=api,
        entity=entity,
        project=project,
        required_tags=list(args.tags),
        required_names=list(args.run_names),
        output_dir=Path(args.output_dir).expanduser(),
        limit=args.limit,
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
    )

    print(
        json.dumps(
            {
                "entity": manifest["entity"],
                "project": manifest["project"],
                "requested_tags": manifest["requested_tags"],
                "requested_run_names": manifest["requested_run_names"],
                "matched_runs": manifest["matched_runs"],
                "exported_runs": manifest["exported_runs"],
                "skipped_runs": manifest["skipped_runs"],
                "dry_run": manifest["dry_run"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
