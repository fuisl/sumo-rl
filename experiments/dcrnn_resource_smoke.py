from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any


_CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
)

for env_var in _CPU_THREAD_ENV_VARS:
    os.environ.setdefault(env_var, "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "SUMO_HOME" in os.environ:
    sumo_tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if sumo_tools not in sys.path:
        sys.path.append(sumo_tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

try:
    from hydra import compose, initialize_config_dir
except ImportError as exc:
    raise SystemExit(
        "Hydra is required for experiments/dcrnn_resource_smoke.py. "
        'Install the experiment extras first, for example: pip install -e ".[experiments]"'
    ) from exc

from omegaconf import DictConfig, OmegaConf

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required for experiments/dcrnn_resource_smoke.py. "
        'Install the RLlib extras first, for example: pip install -e ".[rllib]"'
    ) from exc

from sumo_rl.experiments.rllib_runner import (
    _apply_cpu_thread_limit,
    _build_algorithm_config,
    _build_eval_env,
    _clear_ray_auto_discovery_state,
    _compute_single_action,
    _cuda_visible_devices_env,
    _is_auto_ray_address,
    _is_existing_ray_address,
    _optional_positive_int,
    _policy_id_for_agent,
    _policy_mode,
    _print_ray_resource_summary,
    _ray_address,
    _rllib_runtime_params,
    _train_algorithm,
    normalize_algorithm_kind,
)
from sumo_rl.agents.rllib_common import episode_steps as _episode_steps_from_cfg
from sumo_rl.experiments.runner import _resolve_num_gpus

try:
    from gymnasium import spaces as gym_spaces
except ImportError:  # pragma: no cover - gymnasium is already required at runtime
    gym_spaces = None


@dataclass
class GpuSample:
    timestamp: float
    memory_used_mb: float
    utilization_pct: float


@dataclass
class EpisodeMarker:
    episode_index: int
    timestamp: float
    train_episode_index: float | None
    train_env_step: float | None
    shared_stats: dict[str, float]
    cuda_metrics: dict[str, float | None]


class GpuSampler:
    def __init__(self, *, gpu_index: int, interval_seconds: float, output_path: Path):
        self.gpu_index = gpu_index
        self.interval_seconds = interval_seconds
        self.output_path = output_path
        self._samples: list[GpuSample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._probe() is not None
        return self._available

    def start(self) -> None:
        if not self.is_available():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
        self._thread = None
        self._write_samples()

    def summary(self, baseline_mb: float | None) -> dict[str, float | None]:
        if not self._samples:
            return {
                "gpu_baseline_memory_mb": baseline_mb,
                "gpu_peak_memory_mb": None,
                "gpu_peak_memory_delta_mb": None,
                "gpu_average_utilization_pct": None,
                "gpu_sample_count": 0.0,
            }

        peak_memory = max(sample.memory_used_mb for sample in self._samples)
        avg_utilization = statistics.fmean(sample.utilization_pct for sample in self._samples)
        peak_delta = None if baseline_mb is None else max(0.0, peak_memory - baseline_mb)
        return {
            "gpu_baseline_memory_mb": baseline_mb,
            "gpu_peak_memory_mb": peak_memory,
            "gpu_peak_memory_delta_mb": peak_delta,
            "gpu_average_utilization_pct": avg_utilization,
            "gpu_sample_count": float(len(self._samples)),
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._probe()
            if sample is not None:
                self._samples.append(sample)
            self._stop_event.wait(self.interval_seconds)

    def _probe(self) -> GpuSample | None:
        command = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None

        raw_line = completed.stdout.strip().splitlines()
        if not raw_line:
            return None
        try:
            memory_raw, utilization_raw = [part.strip() for part in raw_line[0].split(",", maxsplit=1)]
            return GpuSample(
                timestamp=time.time(),
                memory_used_mb=float(memory_raw),
                utilization_pct=float(utilization_raw),
            )
        except (ValueError, IndexError):
            return None

    def _write_samples(self) -> None:
        if not self._samples:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "memory_used_mb", "utilization_pct"])
            for sample in self._samples:
                writer.writerow([sample.timestamp, sample.memory_used_mb, sample.utilization_pct])


def _default_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return ROOT / "outputs" / "dcrnn_resource_smoke" / timestamp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run short RLlib smoke jobs for DCRNN variants and record approximate "
            "resource-usage metrics such as GPU memory, GPU utilization, training "
            "wall time, inference latency, and parameter count."
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["ppo_dcrnn_mlp", "ppo_dcrnn_shared_mlp"],
        help="RLlib algorithm kinds to benchmark.",
    )
    parser.add_argument("--scenario", default="resco_grid4x4", help="Hydra scenario config name.")
    parser.add_argument("--episodes", type=int, default=2, help="Smoke-run episode count per variant.")
    parser.add_argument("--episode-seconds", type=int, default=300, help="SUMO episode horizon in seconds.")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=256,
        help="Smoke override for algorithm.params.train_batch_size_per_learner.",
    )
    parser.add_argument(
        "--sgd-minibatch-size",
        type=int,
        default=64,
        help="Smoke override for algorithm.params.sgd_minibatch_size when the variant uses it.",
    )
    parser.add_argument(
        "--num-sgd-iter",
        type=int,
        default=1,
        help="Smoke override for algorithm.params.num_sgd_iter when the variant uses it.",
    )
    parser.add_argument(
        "--inference-repeats",
        type=int,
        default=50,
        help="How many repeated policy-forward passes to use for the inference probe.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
        help="Seconds between GPU samples while training is running.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override for the smoke runs.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Physical GPU index to sample with nvidia-smi after CUDA visibility filtering.",
    )
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=None,
        help="Optional override for resources.ray_num_cpus.",
    )
    parser.add_argument(
        "--native-num-threads",
        type=int,
        default=1,
        help="Override for resources.native_num_threads.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help="Directory where per-variant artifacts and summary files will be written.",
    )
    return parser.parse_args()


def _compose_cfg(args: argparse.Namespace, variant: str, run_dir: Path) -> DictConfig:
    overrides = [
        "logging=disabled",
        f"scenario={args.scenario}",
        f"algorithm={variant}",
        "experiment.name=dcrnn_resource_smoke",
        f"experiment.group=resource_usage_{args.scenario}",
        f"experiment.episodes={args.episodes}",
        f"experiment.episode_seconds={args.episode_seconds}",
        "experiment.validation_interval_episodes=0",
        "logging.save_final_model=false",
        "logging.save_checkpoints=false",
        "logging.save_best_validation_checkpoints=false",
        "logging.validation_log_action_shares=false",
        "logging.validation_log_action_timelines=false",
        "logging.validation_log_phase_queues=false",
        "logging.validation_log_tripinfo_distributions=false",
        "logging.log_final_traffic_metrics=false",
        "logging.log_traffic_metrics_during_training=false",
        f"algorithm.params.train_batch_size_per_learner={args.train_batch_size}",
        f"algorithm.params.sgd_minibatch_size={args.sgd_minibatch_size}",
        f"algorithm.params.num_sgd_iter={args.num_sgd_iter}",
        f"hydra.run.dir={run_dir.as_posix()}",
    ]
    if args.cuda_visible_devices is not None:
        overrides.append(f"resources.cuda_visible_devices={args.cuda_visible_devices}")
    if args.ray_num_cpus is not None:
        overrides.append(f"resources.ray_num_cpus={args.ray_num_cpus}")
    if args.native_num_threads is not None:
        overrides.append(f"resources.native_num_threads={args.native_num_threads}")

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="rllib", overrides=overrides)


def _count_unique_trainable_parameters(module: Any) -> int:
    if module is None or not hasattr(module, "parameters"):
        return 0

    total = 0
    seen: set[tuple[int, int]] = set()
    for parameter in module.parameters():
        if not getattr(parameter, "requires_grad", False):
            continue
        key = (int(parameter.data_ptr()), int(parameter.numel()))
        if key in seen:
            continue
        seen.add(key)
        total += int(parameter.numel())
    return total


def _count_unique_trainable_parameters_from_modules(*modules: Any) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    for module in modules:
        if module is None or not hasattr(module, "parameters"):
            continue
        for parameter in module.parameters():
            if not getattr(parameter, "requires_grad", False):
                continue
            key = (int(parameter.data_ptr()), int(parameter.numel()))
            if key in seen:
                continue
            seen.add(key)
            total += int(parameter.numel())
    return total


def _collect_algo_modules(
    algo: Any,
    cfg: DictConfig,
    run_dir: Path,
    algorithm_kind: str,
) -> list[Any]:
    modules: list[Any] = []
    seen: set[int] = set()

    def _add(candidate: Any) -> None:
        if candidate is None:
            return
        if id(candidate) in seen:
            return
        seen.add(id(candidate))
        modules.append(candidate)

    get_module = getattr(algo, "get_module", None)
    if callable(get_module):
        try:
            _add(get_module())
        except Exception:
            pass

    policy_ids: set[str] = set()
    env = None
    try:
        env = _build_eval_env(
            cfg,
            run_dir,
            seed=int(cfg.experiment.seed),
            algorithm_kind=normalize_algorithm_kind(algorithm_kind),
            policy_mode=_policy_mode(cfg),
        )
        reset_result = env.reset(seed=int(cfg.experiment.seed))
        observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        for agent_id in (observations or {}).keys():
            policy_ids.add(_policy_id_for_agent(str(agent_id), _policy_mode(cfg)))
    except Exception:
        pass
    finally:
        if env is not None:
            env.close()

    if callable(get_module):
        for policy_id in sorted(policy_ids):
            try:
                _add(get_module(policy_id))
            except Exception:
                continue

    get_policy = getattr(algo, "get_policy", None)
    if callable(get_policy):
        candidate_policy_ids = list(sorted(policy_ids)) or [None]
        for policy_id in candidate_policy_ids:
            try:
                policy = get_policy(policy_id) if policy_id is not None else get_policy()
            except Exception:
                policy = None
            if policy is None:
                continue
            for attr_name in ("model", "module"):
                _add(getattr(policy, attr_name, None))

    return modules


def _parameter_breakdown(module: Any) -> dict[str, float]:
    metrics = {
        "parameter_encoder_count": 0.0,
        "parameter_actor_count": 0.0,
        "parameter_critic_count": 0.0,
        "parameter_other_count": 0.0,
    }
    if module is None:
        return metrics

    if isinstance(module, (list, tuple)):
        modules = [item for item in module if item is not None]
        if not modules:
            return metrics
        metrics["parameter_encoder_count"] = float(
            _count_unique_trainable_parameters_from_modules(
                *(getattr(item, "backbone", None) for item in modules),
                *(getattr(item, "_shared_backbone_ref", None) for item in modules),
            )
        )
        metrics["parameter_actor_count"] = float(
            _count_unique_trainable_parameters_from_modules(
                *(getattr(item, "policy_head", None) for item in modules)
            )
        )
        metrics["parameter_critic_count"] = float(
            _count_unique_trainable_parameters_from_modules(
                *(getattr(item, "value_head", None) for item in modules)
            )
        )
        total = float(_count_unique_trainable_parameters_from_modules(*modules))
        accounted = (
            metrics["parameter_encoder_count"]
            + metrics["parameter_actor_count"]
            + metrics["parameter_critic_count"]
        )
        metrics["parameter_other_count"] = max(0.0, total - accounted)
        return metrics

    child_modules = getattr(module, "_rl_modules", None)
    if isinstance(child_modules, dict) and child_modules:
        shared_backbone = getattr(module, "shared_backbone", None)
        if shared_backbone is not None:
            metrics["parameter_encoder_count"] = float(
                _count_unique_trainable_parameters_from_modules(shared_backbone)
            )
            metrics["parameter_actor_count"] = float(
                _count_unique_trainable_parameters_from_modules(
                    *(getattr(child, "policy_head", None) for child in child_modules.values())
                )
            )
            metrics["parameter_critic_count"] = float(
                _count_unique_trainable_parameters_from_modules(
                    *(getattr(child, "value_head", None) for child in child_modules.values())
                )
            )
        else:
            metrics["parameter_encoder_count"] = float(
                _count_unique_trainable_parameters_from_modules(
                    *(getattr(child, "backbone", None) for child in child_modules.values())
                )
            )
            metrics["parameter_actor_count"] = float(
                _count_unique_trainable_parameters_from_modules(
                    *(getattr(child, "policy_head", None) for child in child_modules.values())
                )
            )
            metrics["parameter_critic_count"] = float(
                _count_unique_trainable_parameters_from_modules(
                    *(getattr(child, "value_head", None) for child in child_modules.values())
                )
            )

    total = float(_count_unique_trainable_parameters(module))
    accounted = (
        metrics["parameter_encoder_count"]
        + metrics["parameter_actor_count"]
        + metrics["parameter_critic_count"]
    )
    metrics["parameter_other_count"] = max(0.0, total - accounted)
    return metrics


def _module_device(module: Any) -> torch.device:
    if isinstance(module, (list, tuple)):
        for item in module:
            device = _module_device(item)
            if device.type == "cuda":
                return device
        for item in module:
            device = _module_device(item)
            if device is not None:
                return device
        return torch.device("cpu")
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _used_cuda(module: Any) -> bool:
    return _module_device(module).type == "cuda"


def _torch_cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _torch_cuda_reset_peak_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_memory_snapshot(module: Any, *, label: str) -> dict[str, float | None]:
    device = _module_device(module)
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            f"cuda_{label}_memory_allocated_mb": None,
            f"cuda_{label}_memory_reserved_mb": None,
            f"cuda_{label}_max_memory_allocated_mb": None,
            f"cuda_{label}_max_memory_reserved_mb": None,
        }
    return {
        f"cuda_{label}_memory_allocated_mb": float(torch.cuda.memory_allocated(device) / float(1024**2)),
        f"cuda_{label}_memory_reserved_mb": float(torch.cuda.memory_reserved(device) / float(1024**2)),
        f"cuda_{label}_max_memory_allocated_mb": float(torch.cuda.max_memory_allocated(device) / float(1024**2)),
        f"cuda_{label}_max_memory_reserved_mb": float(torch.cuda.max_memory_reserved(device) / float(1024**2)),
    }


def _observation_storage_diagnostics(cfg: DictConfig, module: Any) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "history_len": None,
        "num_nodes": None,
        "feature_dim": None,
        "observation_dtype_bytes": None,
        "num_policies": None,
        "episode_steps_actual": float(_episode_steps_from_cfg(cfg)),
        "train_batch_size_per_learner_actual": None,
        "minibatch_size_actual": None,
        "per_sample_obs_bytes": None,
        "rollout_obs_bytes": None,
        "minibatch_obs_bytes": None,
    }

    if isinstance(module, (list, tuple)):
        modules = [item for item in module if item is not None]
        if modules:
            metrics["num_policies"] = float(len(modules))
            obs_space = getattr(modules[0], "observation_space", None)
        else:
            obs_space = None
    else:
        child_modules = getattr(module, "_rl_modules", None)
        if isinstance(child_modules, dict) and child_modules:
            metrics["num_policies"] = float(len(child_modules))
            first_module = next(iter(child_modules.values()))
            obs_space = getattr(first_module, "observation_space", None)
        else:
            metrics["num_policies"] = 1.0 if module is not None else None
            obs_space = getattr(module, "observation_space", None)

    raw_sample_bytes = _space_sample_bytes(obs_space)
    if raw_sample_bytes is not None:
        episode_steps_value = max(1, int(metrics["episode_steps_actual"] or 1))
        params = OmegaConf.to_container(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}, resolve=True)
        if not isinstance(params, dict):
            params = {}
        train_batch_size = max(1, int(params.get("train_batch_size_per_learner", episode_steps_value) or episode_steps_value))
        train_batch_size = min(train_batch_size, episode_steps_value)
        minibatch_size = params.get("minibatch_size", params.get("sgd_minibatch_size", train_batch_size))
        minibatch_size = min(max(1, int(minibatch_size or train_batch_size)), train_batch_size)
        num_policies = max(1, int(metrics["num_policies"] or 1))
        metrics.update(
            {
                "train_batch_size_per_learner_actual": float(train_batch_size),
                "minibatch_size_actual": float(minibatch_size),
                "per_sample_obs_bytes": float(raw_sample_bytes),
                "rollout_obs_bytes": float(raw_sample_bytes * episode_steps_value * num_policies),
                "minibatch_obs_bytes": float(raw_sample_bytes * minibatch_size),
            }
        )

    obs_shape = tuple(getattr(obs_space, "shape", ()) or ())
    if len(obs_shape) != 3:
        return metrics

    history_len, num_nodes, feature_dim = (int(dim) for dim in obs_shape)
    dtype = getattr(obs_space, "dtype", None)
    dtype_bytes = getattr(dtype, "itemsize", None)
    if dtype_bytes is None:
        dtype_bytes = 4

    metrics.update(
        {
            "history_len": float(history_len),
            "num_nodes": float(num_nodes),
            "feature_dim": float(feature_dim),
            "observation_dtype_bytes": float(dtype_bytes),
        }
    )
    return metrics


def _shared_forward_diagnostics(module: Any, *, label: str) -> dict[str, float]:
    getter = getattr(module, "shared_forward_stats", None)
    if not callable(getter):
        return {}
    stats = getter() or {}
    return {f"shared_forward_{label}_{key}": float(value) for key, value in stats.items()}


def _shared_forward_stats_snapshot(module: Any) -> dict[str, float]:
    getter = getattr(module, "shared_forward_stats", None)
    if not callable(getter):
        return {}
    stats = getter() or {}
    return {str(key): float(value) for key, value in stats.items()}


def _environment_path_diagnostics(algorithm_kind: str) -> dict[str, Any]:
    normalized_kind = normalize_algorithm_kind(algorithm_kind)
    graph_eval_variants = {
        "dqn_dcrnn",
        "dqn_dcrnn_mlp",
        "ppo_dcrnn_mlp",
        "ppo_dcrnn_shared_mlp",
        "sac_dcrnn_actor",
        "sac_dcrnn_actor_mlp",
        "sac_dcrnn_full",
        "sac_dcrnn_full_mlp",
        "sac_dcrnn_shared_mlp",
    }
    uses_graph_eval_env = normalized_kind in graph_eval_variants
    return {
        "env_pipeline": "graph_parallel_pettingzoo" if uses_graph_eval_env else "parallel_pettingzoo",
        "env_base_factory": "build_sumo_parallel_env",
        "env_wrapper": "build_rllib_graph_parallel_env" if uses_graph_eval_env else "build_rllib_parallel_env",
        "env_observation_mode": "graph_history" if uses_graph_eval_env else "default",
    }


def _cuda_episode_snapshot(module: Any) -> dict[str, float | None]:
    snapshot = _cuda_memory_snapshot(module, label="episode")
    return {
        "episode_cuda_memory_allocated_mb": snapshot.get("cuda_episode_memory_allocated_mb"),
        "episode_cuda_memory_reserved_mb": snapshot.get("cuda_episode_memory_reserved_mb"),
        "episode_cuda_max_memory_allocated_mb": snapshot.get("cuda_episode_max_memory_allocated_mb"),
        "episode_cuda_max_memory_reserved_mb": snapshot.get("cuda_episode_max_memory_reserved_mb"),
    }


def _gpu_interval_summary(
    samples: list[GpuSample],
    *,
    start_timestamp: float,
    end_timestamp: float,
    baseline_mb: float | None,
) -> dict[str, float | None]:
    interval_samples = [
        sample
        for sample in samples
        if start_timestamp < float(sample.timestamp) <= end_timestamp
    ]
    if not interval_samples:
        return {
            "episode_gpu_peak_memory_mb": None,
            "episode_gpu_peak_memory_delta_mb": None,
            "episode_gpu_average_utilization_pct": None,
            "episode_gpu_sample_count": 0.0,
        }

    peak_memory = max(sample.memory_used_mb for sample in interval_samples)
    average_utilization = statistics.fmean(sample.utilization_pct for sample in interval_samples)
    peak_delta = None if baseline_mb is None else max(0.0, peak_memory - baseline_mb)
    return {
        "episode_gpu_peak_memory_mb": float(peak_memory),
        "episode_gpu_peak_memory_delta_mb": None if peak_delta is None else float(peak_delta),
        "episode_gpu_average_utilization_pct": float(average_utilization),
        "episode_gpu_sample_count": float(len(interval_samples)),
    }


def _build_episode_rows(
    episode_markers: list[EpisodeMarker],
    *,
    samples: list[GpuSample],
    baseline_mb: float | None,
    training_start_timestamp: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_timestamp = float(training_start_timestamp)
    previous_stats: dict[str, float] = {}
    for marker in episode_markers:
        current_stats = dict(marker.shared_stats)
        episode_hits = max(
            0.0,
            float(current_stats.get("hits", 0.0)) - float(previous_stats.get("hits", 0.0)),
        )
        episode_fallbacks = max(
            0.0,
            float(current_stats.get("fallbacks", 0.0)) - float(previous_stats.get("fallbacks", 0.0)),
        )
        total_attempts = episode_hits + episode_fallbacks
        row: dict[str, Any] = {
            "episode_index": float(marker.episode_index),
            "train_episode_index": marker.train_episode_index,
            "train_env_step": marker.train_env_step,
            "episode_wall_clock_seconds": max(0.0, float(marker.timestamp) - previous_timestamp),
            "episode_shared_forward_hits": episode_hits,
            "episode_shared_forward_fallbacks": episode_fallbacks,
            "episode_shared_forward_hit_rate": (episode_hits / total_attempts) if total_attempts > 0 else None,
        }
        row.update(marker.cuda_metrics)
        row.update(
            _gpu_interval_summary(
                samples,
                start_timestamp=previous_timestamp,
                end_timestamp=float(marker.timestamp),
                baseline_mb=baseline_mb,
            )
        )
        rows.append(row)
        previous_timestamp = float(marker.timestamp)
        previous_stats = current_stats
    return rows


def _episode_average_summary(episode_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not episode_rows:
        return {"episode_row_count": 0.0}

    average_fields = (
        "episode_wall_clock_seconds",
        "episode_gpu_peak_memory_delta_mb",
        "episode_gpu_average_utilization_pct",
        "episode_shared_forward_hits",
        "episode_shared_forward_fallbacks",
        "episode_shared_forward_hit_rate",
        "episode_cuda_max_memory_allocated_mb",
        "episode_cuda_max_memory_reserved_mb",
    )
    summary: dict[str, float] = {"episode_row_count": float(len(episode_rows))}
    for field in average_fields:
        values = [
            float(row[field])
            for row in episode_rows
            if isinstance(row.get(field), (int, float)) and row.get(field) is not None
        ]
        if values:
            summary[f"{field}_mean"] = float(statistics.fmean(values))

    peak_fields = (
        "episode_gpu_peak_memory_delta_mb",
        "episode_cuda_max_memory_allocated_mb",
        "episode_cuda_max_memory_reserved_mb",
    )
    for field in peak_fields:
        values = [
            float(row[field])
            for row in episode_rows
            if isinstance(row.get(field), (int, float)) and row.get(field) is not None
        ]
        if values:
            summary[f"{field}_run_max"] = float(max(values))
    return summary


def _space_sample_bytes(space: Any) -> int | None:
    if space is None or gym_spaces is None:
        return None

    if isinstance(space, gym_spaces.Box):
        shape = tuple(getattr(space, "shape", ()) or ())
        dtype = getattr(space, "dtype", None)
        itemsize = getattr(dtype, "itemsize", None)
        if itemsize is None:
            return None
        element_count = 1
        for dim in shape:
            element_count *= int(dim)
        return int(element_count) * int(itemsize)

    if isinstance(space, gym_spaces.Discrete):
        dtype = getattr(space, "dtype", None)
        itemsize = getattr(dtype, "itemsize", None)
        if itemsize is None:
            itemsize = 8
        return int(itemsize)

    if isinstance(space, gym_spaces.MultiBinary):
        shape = tuple(getattr(space, "shape", ()) or ())
        element_count = 1
        for dim in shape:
            element_count *= int(dim)
        return int(element_count)

    if isinstance(space, gym_spaces.MultiDiscrete):
        dtype = getattr(space, "dtype", None)
        itemsize = getattr(dtype, "itemsize", None)
        if itemsize is None:
            return None
        shape = tuple(getattr(getattr(space, "nvec", None), "shape", ()) or ())
        element_count = 1
        for dim in shape:
            element_count *= int(dim)
        return int(element_count) * int(itemsize)

    if isinstance(space, gym_spaces.Tuple):
        total = 0
        for child in getattr(space, "spaces", ()):
            child_bytes = _space_sample_bytes(child)
            if child_bytes is None:
                return None
            total += int(child_bytes)
        return total

    if isinstance(space, gym_spaces.Dict):
        total = 0
        for child in getattr(space, "spaces", {}).values():
            child_bytes = _space_sample_bytes(child)
            if child_bytes is None:
                return None
            total += int(child_bytes)
        return total

    return None


def _build_algo(cfg: DictConfig, run_dir: Path, algorithm_kind: str):
    config = _build_algorithm_config(cfg, run_dir, algorithm_kind)
    build_algo = getattr(config, "build_algo", None)
    return build_algo() if callable(build_algo) else config.build()


def _init_ray_for_cfg(cfg: DictConfig) -> Any:
    import ray

    params = OmegaConf.to_container(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}, resolve=True)
    if not isinstance(params, dict):
        params = {}
    runtime_params = _rllib_runtime_params(cfg)
    ray_num_cpus = _optional_positive_int(runtime_params.get("ray_num_cpus", 2), setting_name="ray_num_cpus")
    native_num_threads = _optional_positive_int(
        runtime_params.get("native_num_threads", 1),
        setting_name="native_num_threads",
    )
    ray_address = _ray_address(runtime_params.get("ray_address", os.environ.get("RAY_ADDRESS")))
    cpu_thread_env = _apply_cpu_thread_limit(native_num_threads)
    cuda_env = _cuda_visible_devices_env(runtime_params.get("cuda_visible_devices"))
    for env_var, value in cuda_env.items():
        os.environ[env_var] = value
    ray_num_gpus = _resolve_num_gpus(params.get("ray_num_gpus", params.get("num_gpus_per_learner", "auto")))
    runtime_env_vars = dict(cpu_thread_env)
    runtime_env_vars.update(cuda_env)
    local_ray_startup = not _is_existing_ray_address(ray_address)
    if local_ray_startup:
        _clear_ray_auto_discovery_state(ray)

    ray_init_kwargs: dict[str, Any] = {
        "ignore_reinit_error": True,
        "log_to_driver": False,
    }
    if local_ray_startup:
        ray_init_kwargs["address"] = "local"
        ray_init_kwargs["include_dashboard"] = False
        ray_init_kwargs["num_gpus"] = ray_num_gpus
        if ray_num_cpus is not None:
            ray_init_kwargs["num_cpus"] = ray_num_cpus
    elif ray_address is not None:
        ray_init_kwargs["address"] = ray_address
    if runtime_env_vars:
        ray_init_kwargs["runtime_env"] = {"env_vars": runtime_env_vars}

    try:
        ray.init(**ray_init_kwargs)
    except ConnectionError:
        if not _is_auto_ray_address(ray_address):
            raise
        ray.shutdown()
        _clear_ray_auto_discovery_state(ray)
        fallback_kwargs: dict[str, Any] = {
            "address": "local",
            "ignore_reinit_error": True,
            "log_to_driver": False,
            "include_dashboard": False,
            "num_gpus": ray_num_gpus,
        }
        if ray_num_cpus is not None:
            fallback_kwargs["num_cpus"] = ray_num_cpus
        if runtime_env_vars:
            fallback_kwargs["runtime_env"] = {"env_vars": runtime_env_vars}
        ray.init(**fallback_kwargs)

    _print_ray_resource_summary(ray)
    return ray


def _measure_inference(
    algo: Any,
    cfg: DictConfig,
    run_dir: Path,
    algorithm_kind: str,
    repeats: int,
) -> dict[str, float]:
    algorithm_kind = normalize_algorithm_kind(algorithm_kind)
    env = _build_eval_env(
        cfg,
        run_dir,
        seed=int(cfg.experiment.seed),
        algorithm_kind=algorithm_kind,
        policy_mode=_policy_mode(cfg),
    )
    try:
        reset_result = env.reset(seed=int(cfg.experiment.seed))
        observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        if not observations:
            return {
                "inference_joint_decision_ms": 0.0,
                "inference_agent_action_ms": 0.0,
                "inference_agent_count": 0.0,
            }

        module = algo.get_module() if hasattr(algo, "get_module") else None
        resolved_modules = _collect_algo_modules(algo, cfg, run_dir, algorithm_kind)
        device = _module_device(resolved_modules or module)
        agent_ids = list(observations.keys())
        reset_shared_stats = getattr(module, "reset_shared_forward_stats", None)
        if callable(reset_shared_stats):
            reset_shared_stats()
        _torch_cuda_reset_peak_if_needed(device)

        for _ in range(5):
            for agent_id in agent_ids:
                _compute_single_action(
                    algo,
                    observations[agent_id],
                    policy_id=_policy_id_for_agent(agent_id, _policy_mode(cfg)),
                    algorithm_kind=algorithm_kind,
                )
        _torch_cuda_sync_if_needed(device)
        warmup_metrics = _cuda_memory_snapshot(resolved_modules or module, label="after_warmup_inference")

        start = time.perf_counter()
        total_agent_actions = 0
        for _ in range(repeats):
            for agent_id in agent_ids:
                _compute_single_action(
                    algo,
                    observations[agent_id],
                    policy_id=_policy_id_for_agent(agent_id, _policy_mode(cfg)),
                    algorithm_kind=algorithm_kind,
                )
                total_agent_actions += 1
        _torch_cuda_sync_if_needed(device)
        elapsed = time.perf_counter() - start

        metrics = {
            "inference_joint_decision_ms": (elapsed / max(1, repeats)) * 1000.0,
            "inference_agent_action_ms": (elapsed / max(1, total_agent_actions)) * 1000.0,
            "inference_agent_count": float(len(agent_ids)),
        }
        metrics.update(warmup_metrics)
        metrics.update(_shared_forward_diagnostics(module, label="inference"))
        return metrics
    finally:
        env.close()


def _probe_variant(
    cfg: DictConfig,
    run_dir: Path,
    algorithm_kind: str,
    repeats: int,
) -> dict[str, float]:
    ray = _init_ray_for_cfg(cfg)
    algo = None
    try:
        algo = _build_algo(cfg, run_dir, algorithm_kind)
        modules = _collect_algo_modules(algo, cfg, run_dir, algorithm_kind)
        metrics = {
            "parameter_count": float(_count_unique_trainable_parameters_from_modules(*modules)),
            "used_cuda": float(1.0 if _used_cuda(modules) else 0.0),
        }
        metrics.update(_parameter_breakdown(modules))
        metrics.update(_observation_storage_diagnostics(cfg, modules))
        metrics.update(_cuda_memory_snapshot(modules, label="post_build"))
        _torch_cuda_reset_peak_if_needed(_module_device(modules))
        metrics.update(_measure_inference(algo, cfg, run_dir, algorithm_kind, repeats))
        return metrics
    finally:
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
        ray.shutdown()


def _train_variant(
    cfg: DictConfig,
    run_dir: Path,
    algorithm_kind: str,
    gpu_index: int,
    sample_interval: float,
) -> dict[str, Any]:
    ray = _init_ray_for_cfg(cfg)
    algo = None
    metrics: dict[str, float | None] = {}
    episode_rows: list[dict[str, Any]] = []
    wall_clock_seconds = 0.0
    gpu_log_path = run_dir / "resource_usage" / "gpu_samples.csv"
    sampler = GpuSampler(gpu_index=gpu_index, interval_seconds=sample_interval, output_path=gpu_log_path)
    baseline_sample = sampler._probe() if sampler.is_available() else None
    baseline_mb = None if baseline_sample is None else baseline_sample.memory_used_mb
    episode_markers: list[EpisodeMarker] = []
    training_start_timestamp = time.time()

    try:
        start = time.perf_counter()
        sampler.start()
        algo = _build_algo(cfg, run_dir, algorithm_kind)
        module = algo.get_module() if hasattr(algo, "get_module") else None
        resolved_modules = _collect_algo_modules(algo, cfg, run_dir, algorithm_kind)
        metrics.update(_cuda_memory_snapshot(resolved_modules or module, label="train_post_build"))
        metrics["used_cuda"] = float(1.0 if _used_cuda(resolved_modules or module) else 0.0)
        reset_shared_stats = getattr(module, "reset_shared_forward_stats", None)
        if callable(reset_shared_stats):
            reset_shared_stats()
        _torch_cuda_reset_peak_if_needed(_module_device(resolved_modules or module))
        first_iteration_recorded = False
        recorded_episode_indices: set[int] = set()

        def _record_episode_marker(emitted_metrics, step: int) -> None:
            if int(step) in recorded_episode_indices:
                return
            recorded_episode_indices.add(int(step))
            episode_markers.append(
                EpisodeMarker(
                    episode_index=int(step),
                    timestamp=time.time(),
                    train_episode_index=(
                        float(emitted_metrics.get("train/episode_index"))
                        if isinstance(emitted_metrics.get("train/episode_index"), (int, float))
                        else None
                    ),
                    train_env_step=(
                        float(emitted_metrics.get("train/env_step"))
                        if isinstance(emitted_metrics.get("train/env_step"), (int, float))
                        else None
                    ),
                    shared_stats=_shared_forward_stats_snapshot(module),
                    cuda_metrics=_cuda_episode_snapshot(resolved_modules or module),
                )
            )
            _torch_cuda_reset_peak_if_needed(_module_device(resolved_modules or module))

        def _capture_train_metrics(emitted_metrics, step):
            nonlocal first_iteration_recorded
            if not first_iteration_recorded:
                first_iteration_recorded = True
                metrics.update(_cuda_memory_snapshot(resolved_modules or module, label="after_first_train_iteration"))
                metrics.update(_shared_forward_diagnostics(module, label="train_first_iteration"))
            _record_episode_marker(emitted_metrics, int(step))

        _train_algorithm(
            algo,
            cfg,
            algorithm_kind,
            emit_metrics=_capture_train_metrics,
            validate=lambda metrics, step: {},
        )
        wall_clock_seconds = time.perf_counter() - start
        metrics.update(_cuda_memory_snapshot(resolved_modules or module, label="after_training_end"))
        metrics.update(_shared_forward_diagnostics(module, label="train_total"))
    finally:
        sampler.stop()
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
        ray.shutdown()

    episode_rows = _build_episode_rows(
        episode_markers,
        samples=list(sampler._samples),
        baseline_mb=baseline_mb,
        training_start_timestamp=training_start_timestamp,
    )
    _write_summary_csv(run_dir / "resource_usage" / "episode_rows.csv", episode_rows)
    _write_json(run_dir / "resource_usage" / "episode_rows.json", {"rows": episode_rows})

    metrics["wall_clock_training_seconds"] = wall_clock_seconds
    metrics.update(sampler.summary(baseline_mb))
    metrics.update(_episode_average_summary(episode_rows))
    return {
        "summary_metrics": metrics,
        "episode_rows": episode_rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_variant(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    normalized_variant = normalize_algorithm_kind(variant)
    run_dir = args.output_root / normalized_variant
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _compose_cfg(args, normalized_variant, run_dir)
    result: dict[str, Any] = {
        "variant": normalized_variant,
        "scenario": args.scenario,
        "episodes": float(args.episodes),
        "episode_seconds": float(args.episode_seconds),
        "train_batch_size_per_learner": float(args.train_batch_size),
        "sgd_minibatch_size": float(args.sgd_minibatch_size),
        "num_sgd_iter": float(args.num_sgd_iter),
        "status": "ok",
        "run_dir": str(run_dir),
    }
    result.update(_environment_path_diagnostics(normalized_variant))

    try:
        result.update(_probe_variant(cfg, run_dir, normalized_variant, args.inference_repeats))
        train_result = _train_variant(
            cfg,
            run_dir,
            normalized_variant,
            gpu_index=args.gpu_index,
            sample_interval=args.sample_interval,
        )
        result.update(train_result.get("summary_metrics", {}))
        result["episode_rows_path"] = str(run_dir / "resource_usage" / "episode_rows.csv")
        result["episode_rows_json_path"] = str(run_dir / "resource_usage" / "episode_rows.json")
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _write_json(run_dir / "resource_usage" / "summary.json", result)
    return result


def _print_summary(rows: list[dict[str, Any]]) -> None:
    headers = [
        "variant",
        "status",
        "parameter_count",
        "parameter_encoder_count",
        "parameter_actor_count",
        "parameter_critic_count",
        "parameter_other_count",
        "episode_row_count",
        "episode_wall_clock_seconds_mean",
        "episode_gpu_peak_memory_delta_mb_mean",
        "wall_clock_training_seconds",
        "inference_joint_decision_ms",
        "inference_agent_action_ms",
        "gpu_peak_memory_delta_mb",
        "gpu_average_utilization_pct",
    ]
    print()
    print("DCRNN resource smoke summary")
    print("-" * 120)
    print(" | ".join(headers))
    print("-" * 120)
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        print(" | ".join(values))


def main() -> None:
    args = _parse_args()
    args.output_root = Path(args.output_root).resolve()
    rows = [_run_variant(args, variant) for variant in args.variants]
    _write_summary_csv(args.output_root / "resource_usage_summary.csv", rows)
    _write_json(args.output_root / "resource_usage_summary.json", {"rows": rows})
    _print_summary(rows)
    print()
    print(f"Artifacts written to {args.output_root}")


if __name__ == "__main__":
    main()
