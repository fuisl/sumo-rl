from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.fspath)).resolve()
        parts = path.parts
        if "tests" not in parts:
            continue

        tests_index = parts.index("tests")
        relative_parts = parts[tests_index + 1 :]
        if not relative_parts:
            continue

        top_level = relative_parts[0]
        if top_level == "integration_local":
            item.add_marker(pytest.mark.local_heavy)
        elif top_level in {"models", "runner"}:
            item.add_marker(pytest.mark.research_heavy)
