"""Environment variable helpers used before SUMO imports."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parents[2]


def resolve_env_file(explicit_path: str = "") -> Path | None:
    """Resolve a dotenv file from an explicit path, cwd, or the repo root."""

    candidates: list[Path] = []
    if explicit_path:
        path = Path(explicit_path).expanduser()
        candidates.append(path if path.is_absolute() else Path.cwd() / path)
        candidates.append(repo_root() / path.name)

    cwd = Path.cwd().resolve()
    candidates.extend(parent / ".env" for parent in (cwd, *cwd.parents))
    candidates.append(repo_root() / ".env")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def load_env_file(env_path: Path | None) -> None:
    """Load environment variables from a dotenv file when one is available."""

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


def load_repo_env(explicit_path: str = "") -> Path | None:
    """Load the closest repo dotenv file without overriding existing env vars."""

    env_path = resolve_env_file(explicit_path)
    load_env_file(env_path)
    return env_path
