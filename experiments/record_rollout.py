from __future__ import annotations

import argparse
import tempfile
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageGrab


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record an MP4 rollout for RLlib checkpoints or static controllers.")
    parser.add_argument(
        "--controller",
        choices=("rllib", "fixed_time", "static_max_pressure"),
        required=True,
        help="Which policy/controller source to record.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4 path. Defaults under the run directory or outputs/recordings/ for static controllers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Evaluation seed. Defaults to experiment.seed from the resolved config.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Output video frames per second.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Keep every Nth environment step in the video.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on rollout steps for quick smoke recordings.",
    )
    parser.add_argument(
        "--use-gui",
        action="store_true",
        help="Also request SUMO GUI mode while recording. Frame capture still uses rgb_array rendering.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
        help="Frame width for rgb_array capture.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="Frame height for rgb_array capture.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Hydra run directory for RLlib checkpoint playback.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint directory to restore for RLlib playback.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Hydra config name for static controllers. Defaults to fixed_time or static_max_pressure entrypoints.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional Hydra scenario override for static controllers, for example resco_grid4x4.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override for static controllers. Repeatable, for example --override env.kwargs.num_seconds=600.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be a positive integer.")
    if args.frame_skip <= 0:
        raise ValueError("--frame-skip must be a positive integer.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be a positive integer when provided.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive integers.")
    if args.controller == "rllib":
        if not args.run_dir or not args.checkpoint:
            raise ValueError("--run-dir and --checkpoint are required when --controller rllib is used.")
    else:
        if args.run_dir or args.checkpoint:
            raise ValueError("--run-dir and --checkpoint are only valid with --controller rllib.")


def _prepare_video_config(cfg: Any, *, use_gui: bool, width: int, height: int) -> Any:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    env_cfg = getattr(cfg, "env", None)
    kwargs = getattr(env_cfg, "kwargs", None)
    if kwargs is None:
        cfg.env.kwargs = {}
        kwargs = cfg.env.kwargs
    del use_gui
    # Record from the real SUMO GUI window via screenshots so this works on
    # Windows without the Xvfb-based rgb_array path.
    kwargs.render_mode = None
    kwargs.use_gui = True
    kwargs.virtual_display = [int(width), int(height)]
    return cfg


def _ensure_rgb_frame(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"Expected HxWx3/4 frame array, got shape {array.shape!r}.")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


class _Cv2VideoWriter:
    def __init__(self, output_path: Path, *, fps: int, width: int, height: int) -> None:
        import cv2

        self._cv2 = cv2
        self._writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(width), int(height)),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"OpenCV could not open the MP4 writer for {output_path}.")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))

    def release(self) -> None:
        self._writer.release()


class _ImageioVideoWriter:
    def __init__(self, output_path: Path, *, fps: int, width: int, height: int) -> None:
        del width, height
        import imageio.v2 as imageio

        self._writer = imageio.get_writer(str(output_path), fps=int(fps), format="FFMPEG")

    def write(self, frame: np.ndarray) -> None:
        self._writer.append_data(frame)

    def release(self) -> None:
        self._writer.close()


def _open_video_writer(output_path: Path, *, fps: int, width: int, height: int):
    errors: list[str] = []

    try:
        return _Cv2VideoWriter(output_path, fps=fps, width=width, height=height)
    except Exception as exc:
        errors.append(f"OpenCV backend unavailable: {type(exc).__name__}: {exc}")

    try:
        return _ImageioVideoWriter(output_path, fps=fps, width=width, height=height)
    except Exception as exc:
        errors.append(f"imageio/FFMPEG backend unavailable: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not create an MP4 writer. Install either OpenCV or imageio with imageio-ffmpeg. "
        + " ".join(errors)
    )


def _capture_gui_screenshot(base_env: Any, screenshot_path: Path, *, width: int, height: int) -> np.ndarray:
    sumo = getattr(base_env, "sumo", None)
    if sumo is None or not hasattr(sumo, "gui"):
        raise RuntimeError("SUMO GUI connection is unavailable, so a screenshot frame could not be captured.")

    view_id = "View #0"
    sumo.gui.screenshot(view_id, str(screenshot_path), width=int(width), height=int(height))
    last_error: Exception | None = None
    for attempt in range(20):
        try:
            with Image.open(screenshot_path) as image:
                return _ensure_rgb_frame(image.convert("RGB"))
        except FileNotFoundError as exc:
            last_error = exc
            time.sleep(0.1 * (attempt + 1))
    desktop_error: Exception | None = None
    try:
        desktop = ImageGrab.grab(all_screens=True)
        if desktop is not None:
            if width > 0 and height > 0:
                desktop = desktop.resize((int(width), int(height)))
            return _ensure_rgb_frame(desktop.convert("RGB"))
    except Exception as exc:
        desktop_error = exc
    message = f"SUMO GUI screenshot was not written to disk: {screenshot_path}"
    if desktop_error is not None:
        message += f". Desktop capture fallback also failed: {type(desktop_error).__name__}: {desktop_error}"
    raise RuntimeError(message) from last_error


def _capture_frame(env: Any, *, screenshot_path: Path, width: int, height: int) -> np.ndarray:
    from sumo_rl.experiments.rllib_runner import _resolve_sumo_base_env

    base_env = _resolve_sumo_base_env(env)
    render_mode = getattr(base_env, "render_mode", None)
    if render_mode == "rgb_array":
        frame = base_env.render()
        if frame is not None:
            return _ensure_rgb_frame(frame)
    return _capture_gui_screenshot(base_env, screenshot_path, width=width, height=height)


def _done_from_step_result(step_result: Any) -> bool:
    if not isinstance(step_result, tuple):
        raise ValueError("Unexpected environment step result type while recording rollout.")
    if len(step_result) == 5:
        _, _, terminated, truncated, _ = step_result
        if isinstance(terminated, dict):
            return bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        return bool(terminated or truncated)
    if len(step_result) == 4:
        _, _, dones, _ = step_result
        if isinstance(dones, dict):
            return bool(dones.get("__all__", False))
        return bool(dones)
    raise ValueError(f"Unexpected environment step tuple length: {len(step_result)}")


def _reward_from_step_result(step_result: Any) -> float:
    rewards = step_result[1]
    if isinstance(rewards, dict):
        return float(sum(float(value) for value in rewards.values()))
    if rewards is None:
        return 0.0
    return float(rewards)


def _rllib_default_output_path(run_dir: Path, seed: int) -> Path:
    return run_dir / "videos" / f"rllib_rollout_seed{seed}.mp4"


def _static_default_output_path(controller: str, scenario_name: str, seed: int) -> Path:
    safe_scenario = str(scenario_name or "scenario").replace("/", "_")
    return ROOT / "outputs" / "recordings" / f"{controller}__{safe_scenario}__seed{seed}.mp4"


def _build_rllib_actions(
    algo: Any,
    obs: dict[str, Any],
    *,
    agent_ids: list[str],
    algorithm_kind: str,
    policy_mode: str,
) -> dict[str, Any]:
    from sumo_rl.experiments.rllib_runner import _compute_single_action, _policy_id_for_agent

    actions: dict[str, Any] = {}
    for agent_id in agent_ids:
        if agent_id not in obs:
            continue
        actions[agent_id] = _compute_single_action(
            algo,
            obs[agent_id],
            policy_id=_policy_id_for_agent(agent_id, policy_mode),
            algorithm_kind=algorithm_kind,
        )
    return actions


def _record_rollout(
    env: Any,
    *,
    output_path: Path,
    fps: int,
    frame_skip: int,
    max_steps: int | None,
    width: int,
    height: int,
    step_fn,
    reset_kwargs: dict[str, Any] | None = None,
) -> tuple[int, int, float]:
    reset_kwargs = dict(reset_kwargs or {})
    reset_result = env.reset(**reset_kwargs)
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sumo_rollout_") as temp_dir:
        screenshot_path = Path(temp_dir) / "frame.png"
        writer = None
        total_reward = 0.0
        step_count = 0
        written_frames = 0

        try:
            done = False
            while not done:
                obs, step_result = step_fn(obs)
                step_count += 1
                total_reward += _reward_from_step_result(step_result)

                if frame_skip <= 1 or step_count % frame_skip == 0:
                    frame = _capture_frame(
                        env,
                        screenshot_path=screenshot_path,
                        width=width,
                        height=height,
                    )
                    if writer is None:
                        frame_height, frame_width = frame.shape[:2]
                        writer = _open_video_writer(output_path, fps=fps, width=frame_width, height=frame_height)
                    writer.write(frame)
                    written_frames += 1

                done = _done_from_step_result(step_result)
                if max_steps is not None and step_count >= max_steps:
                    done = True
        finally:
            if writer is not None:
                writer.release()

    return step_count, written_frames, total_reward


def _load_static_cfg(args: argparse.Namespace) -> Any:
    from hydra import compose, initialize_config_dir

    config_dir = str((ROOT / "configs").resolve())
    config_name = args.config_name or ("fixed_time" if args.controller == "fixed_time" else "static_max_pressure")
    overrides = list(args.override or [])
    if args.scenario:
        overrides.append(f"scenario={args.scenario}")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _record_rllib_rollout(args: argparse.Namespace) -> Path:
    run_dir = Path(str(args.run_dir)).resolve()
    checkpoint_path = Path(str(args.checkpoint)).resolve()

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    cfg = _prepare_video_config(cfg, use_gui=args.use_gui, width=args.width, height=args.height)
    algorithm_kind = str(cfg.algorithm.kind)

    from sumo_rl.agents.rllib_common import plain_dict as _plain_dict, policy_mode as _policy_mode
    from sumo_rl.experiments.rllib_runner import (
        _build_algorithm_config,
        _build_eval_env,
        _possible_agents,
        _restore_checkpoint,
        _sync_env_runner_weights_for_evaluation,
    )

    params = _plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {})
    mode = _policy_mode(params)
    seed = int(args.seed if args.seed is not None else getattr(getattr(cfg, "experiment", None), "seed", 0) or 0)
    output_path = Path(args.output).resolve() if args.output else _rllib_default_output_path(run_dir, seed)

    import ray

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_gpus=0)
    algo = None
    eval_env = None
    try:
        algo_config = _build_algorithm_config(cfg, run_dir, algorithm_kind)
        build_algo = getattr(algo_config, "build_algo", None)
        algo = build_algo() if callable(build_algo) else algo_config.build()

        _restore_checkpoint(algo, checkpoint_path)
        _sync_env_runner_weights_for_evaluation(algo)
        eval_env = _build_eval_env(cfg, run_dir, seed, algorithm_kind=algorithm_kind, policy_mode=mode)
        agent_ids = [str(agent_id) for agent_id in _possible_agents(eval_env) if not str(agent_id).startswith("__")]

        def _step_fn(obs):
            actions = _build_rllib_actions(
                algo,
                obs,
                agent_ids=agent_ids,
                algorithm_kind=algorithm_kind,
                policy_mode=mode,
            )
            step_result = eval_env.step(actions)
            next_obs = step_result[0]
            return next_obs, step_result

        step_count, written_frames, total_reward = _record_rollout(
            eval_env,
            output_path=output_path,
            fps=int(args.fps),
            frame_skip=int(args.frame_skip),
            max_steps=args.max_steps,
            width=int(args.width),
            height=int(args.height),
            step_fn=_step_fn,
            reset_kwargs={"seed": seed},
        )
        print(f"Saved rollout video to {output_path}")
        print(f"Controller: rllib")
        print(f"Steps: {step_count}")
        print(f"Frames written: {written_frames}")
        print(f"Total reward: {total_reward:.6f}")
        return output_path
    finally:
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception:
                pass
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
        ray.shutdown()


def _record_static_rollout(args: argparse.Namespace) -> Path:
    cfg = _load_static_cfg(args)
    cfg = _prepare_video_config(cfg, use_gui=args.use_gui, width=args.width, height=args.height)

    from sumo_rl.experiments.runner import _build_env, _get_base_env, _get_run_dir

    scenario_name = str(getattr(getattr(cfg, "scenario", None), "name", "") or "").strip()
    seed = int(args.seed if args.seed is not None else getattr(getattr(cfg, "experiment", None), "seed", 0) or 0)
    run_dir = _get_run_dir()
    output_path = Path(args.output).resolve() if args.output else _static_default_output_path(args.controller, scenario_name, seed)
    env = _build_env(cfg, run_dir, seed=seed)
    base_env = _get_base_env(env)

    policy = None
    if args.controller == "static_max_pressure":
        from sumo_rl.agents.static import MaxPressurePolicy

        policy = MaxPressurePolicy()

    agent_ids = [str(agent_id) for agent_id in getattr(base_env, "ts_ids", [])]

    from sumo_rl.experiments.runner import _fixed_time_step_action

    def _step_fn(_obs):
        if policy is None:
            actions = _fixed_time_step_action(env, base_env, agent_ids)
        else:
            actions = {agent_id: policy.select_action(base_env.traffic_signals[agent_id]) for agent_id in agent_ids}
        step_result = env.step(actions)
        next_obs = step_result[0] if isinstance(step_result, tuple) else None
        return next_obs, step_result

    try:
        step_count, written_frames, total_reward = _record_rollout(
            env,
            output_path=output_path,
            fps=int(args.fps),
            frame_skip=int(args.frame_skip),
            max_steps=args.max_steps,
            width=int(args.width),
            height=int(args.height),
            step_fn=_step_fn,
            reset_kwargs={"seed": seed},
        )
        print(f"Saved rollout video to {output_path}")
        print(f"Controller: {args.controller}")
        print(f"Scenario: {scenario_name}")
        print(f"Steps: {step_count}")
        print(f"Frames written: {written_frames}")
        print(f"Total reward: {total_reward:.6f}")
        return output_path
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    if args.controller == "rllib":
        _record_rllib_rollout(args)
    else:
        _record_static_rollout(args)


if __name__ == "__main__":
    main()
