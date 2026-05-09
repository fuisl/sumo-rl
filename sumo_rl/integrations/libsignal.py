from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class LibSignalSpec:
    agent: str
    network: str
    task: str = "tsc"
    world: str = "sumo"
    dataset: str = "onfly"
    interface: str = "libsumo"
    delay_type: str = "real"
    thread_num: int = 1
    ngpu: str = "-1"
    episodes: int = 1
    steps: int = 3600
    test_steps: int = 3600
    learning_start: int = 1000
    buffer_size: int = 5000
    update_model_rate: int = 1
    update_target_rate: int = 10
    save_rate: int = 1
    debug: bool = False


def _to_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    return Path(str(value)).expanduser().resolve()


def resolve_libsignal_root(raw_root: Any = None) -> Path:
    root = _to_path(raw_root)
    if root is not None:
        return root

    env_root = os.environ.get("LIBSIGNAL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    raise FileNotFoundError(
        "LibSignal root not configured. Set `algorithm.params.libsignal_root` or `LIBSIGNAL_ROOT` to your LibSignal checkout."
    )


def parse_libsignal_dtl_log(log_path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if not log_path.exists():
        return rows

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 9:
                continue

            model, mode, step, travel_time, loss, rewards, queue, delay, throughput = parts[:9]
            row: Dict[str, Any] = {
                "libsignal/model": model,
                "libsignal/mode": mode,
                "libsignal/step": int(float(step)),
                "libsignal/travel_time": float(travel_time),
                "libsignal/loss": float(loss),
                "libsignal/rewards": float(rewards),
                "libsignal/queue": float(queue),
                "libsignal/delay": float(delay),
                "libsignal/throughput": float(throughput),
                "libsignal/raw_line": line,
            }
            rows.append(row)

    return rows


def select_libsignal_trace_row(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("No LibSignal metrics rows were found in the DTL log.")

    for desired_mode in ("TEST", "TRAIN"):
        for row in reversed(rows_list):
            if row.get("libsignal/mode") == desired_mode:
                return row

    return rows_list[-1]


def build_phase5_trace_row(
    summary: Dict[str, Any],
    *,
    run_idx: int,
    run_seed: int,
    experiment_name: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run/index": run_idx,
        "episode/index": 1,
        "run_seed": float(run_seed),
        "algorithm/kind": "libsignal_phase5",
        "backend": "libsignal",
        "phase5/model": summary.get("agent", summary.get("libsignal/model")),
        "phase5/network": summary.get("network", ""),
        "phase5/experiment": experiment_name,
        "phase5/source": "libsignal_dtl_log",
        "phase5/selected_mode": summary.get("libsignal/mode"),
        "phase5/selected_step": float(summary.get("libsignal/step", 0.0)),
        "episode/steps": float(summary.get("libsignal/step", 0.0)),
        "episode/reward": float(summary.get("libsignal/rewards", 0.0)),
    }

    row.update(summary)

    queue_value = float(summary.get("libsignal/queue", 0.0))
    delay_value = float(summary.get("libsignal/delay", 0.0))
    travel_time_value = float(summary.get("libsignal/travel_time", 0.0))

    row.update(
        {
            "resco_avg_delay": delay_value,
            "resco_trip_time": travel_time_value,
            "resco_wait": delay_value,
            "resco_queue": queue_value,
            "resco_max_queue": queue_value,
        }
    )
    return row


def load_phase5_summary(summary_path: Path) -> Dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Phase 5 summary at {summary_path} did not contain an object.")
    return data


def _prepare_subprocess_env(libsignal_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(libsignal_root)]

    sumo_home = env.get("SUMO_HOME")
    if sumo_home:
        pythonpath_parts.append(str(Path(sumo_home) / "tools"))

    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join([part for part in pythonpath_parts if part])
    return env


def run_phase5_libsignal_seed(
    *,
    repo_root: Path,
    libsignal_root: Path,
    output_root: Path,
    summary_path: Path,
    spec: LibSignalSpec,
    run_seed: int,
    run_prefix: str,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "sumo_rl.integrations.libsignal_cli",
        "--libsignal-root",
        str(libsignal_root),
        "--output-root",
        str(output_root),
        "--summary-path",
        str(summary_path),
        "--task",
        spec.task,
        "--agent",
        spec.agent,
        "--world",
        spec.world,
        "--dataset",
        spec.dataset,
        "--network",
        spec.network,
        "--interface",
        spec.interface,
        "--delay-type",
        spec.delay_type,
        "--seed",
        str(run_seed),
        "--prefix",
        run_prefix,
        "--thread-num",
        str(spec.thread_num),
        "--ngpu",
        spec.ngpu,
        "--episodes",
        str(spec.episodes),
        "--steps",
        str(spec.steps),
        "--test-steps",
        str(spec.test_steps),
        "--learning-start",
        str(spec.learning_start),
        "--buffer-size",
        str(spec.buffer_size),
        "--update-model-rate",
        str(spec.update_model_rate),
        "--update-target-rate",
        str(spec.update_target_rate),
        "--save-rate",
        str(spec.save_rate),
    ]

    env = _prepare_subprocess_env(libsignal_root)
    env["CUDA_VISIBLE_DEVICES"] = spec.ngpu

    completed = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "LibSignal phase 5 run failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout:\n{completed.stdout}\n"
            f"Stderr:\n{completed.stderr}"
        )

    if not summary_path.exists():
        raise FileNotFoundError(f"LibSignal summary file was not created: {summary_path}")

    return load_phase5_summary(summary_path)
