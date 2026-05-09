from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sumo_rl.integrations.libsignal import parse_libsignal_dtl_log, select_libsignal_trace_row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a LibSignal seed and export a normalized summary.")
    parser.add_argument("--libsignal-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--task", default="tsc")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--world", default="sumo")
    parser.add_argument("--dataset", default="onfly")
    parser.add_argument("--network", default="sumo4x4")
    parser.add_argument("--interface", default="libsumo")
    parser.add_argument("--delay-type", default="real")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--ngpu", default="-1")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--test-steps", type=int, default=3600)
    parser.add_argument("--learning-start", type=int, default=1000)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--update-model-rate", type=int, default=1)
    parser.add_argument("--update-target-rate", type=int, default=10)
    parser.add_argument("--save-rate", type=int, default=1)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    libsignal_root = Path(args.libsignal_root).expanduser().resolve()
    if not libsignal_root.exists():
        raise FileNotFoundError(f"LibSignal root does not exist: {libsignal_root}")

    sys.path.insert(0, str(libsignal_root))

    from common import interface
    from common.registry import Registry
    from common.utils import build_config, get_output_file_path
    from utils.logger import setup_logging

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.ngpu)

    config_args = argparse.Namespace(
        thread_num=args.thread_num,
        ngpu=str(args.ngpu),
        prefix=args.prefix,
        seed=args.seed,
        debug=False,
        interface=args.interface,
        delay_type=args.delay_type,
        task=args.task,
        agent=args.agent,
        world=args.world,
        network=args.network,
        dataset=args.dataset,
    )

    config, _ = build_config(config_args)
    config["world"]["dir"] = str(Path(args.output_root).expanduser().resolve())
    config["world"]["seed"] = int(args.seed)
    config["world"]["gui"] = False
    config["world"]["saveReplay"] = True
    config["world"]["no_warning"] = True
    config["world"]["rlTrafficLight"] = True

    config["trainer"]["thread"] = int(args.thread_num)
    config["trainer"]["ngpu"] = int(args.ngpu)
    config["trainer"]["episodes"] = int(args.episodes)
    config["trainer"]["steps"] = int(args.steps)
    config["trainer"]["test_steps"] = int(args.test_steps)
    config["trainer"]["learning_start"] = int(args.learning_start)
    config["trainer"]["buffer_size"] = int(args.buffer_size)
    config["trainer"]["update_model_rate"] = int(args.update_model_rate)
    config["trainer"]["update_target_rate"] = int(args.update_target_rate)
    config["trainer"]["test_when_train"] = True

    config["logger"]["save_rate"] = int(args.save_rate)

    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.World_param_Interface(config)
    if config["model"].get("graphic", False):
        world_param = Registry.mapping["world_mapping"]["setting"].param
        if config["command"]["world"] in ["cityflow", "sumo"]:
            roadnet_path = os.path.join(world_param["dir"], world_param["roadnetFile"])
        else:
            roadnet_path = world_param["road_file_addr"]
        interface.Graph_World_Interface(roadnet_path)
    interface.Logger_path_Interface(config)
    os.makedirs(Registry.mapping["logger_mapping"]["path"].path, exist_ok=True)
    interface.Trainer_param_Interface(config)
    interface.ModelAgent_param_Interface(config)

    logger = setup_logging(level=20)
    trainer_name = Registry.mapping["command_mapping"]["setting"].param["task"]
    trainer = Registry.mapping["trainer_mapping"][trainer_name](logger)
    task = Registry.mapping["task_mapping"][trainer_name](trainer)
    task.run()

    output_path = Path(get_output_file_path(config))
    log_dir = output_path / Registry.mapping["logger_mapping"]["setting"].param["log_dir"]
    dtl_files = sorted(log_dir.glob("*_DTL.log"), key=lambda path: path.stat().st_mtime)
    if not dtl_files:
        raise FileNotFoundError(f"No LibSignal DTL log file was written under {log_dir}")

    rows = parse_libsignal_dtl_log(dtl_files[-1])
    selected = select_libsignal_trace_row(rows)
    summary = {
        "backend": "libsignal",
        "agent": args.agent,
        "network": args.network,
        "task": args.task,
        "world": args.world,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "prefix": args.prefix,
        "external_output_path": str(output_path),
        "external_log_file": str(dtl_files[-1]),
        "selected_mode": selected.get("libsignal/mode"),
        "selected_step": selected.get("libsignal/step"),
        "selected_travel_time": selected.get("libsignal/travel_time"),
        "selected_loss": selected.get("libsignal/loss"),
        "selected_rewards": selected.get("libsignal/rewards"),
        "selected_queue": selected.get("libsignal/queue"),
        "selected_delay": selected.get("libsignal/delay"),
        "selected_throughput": selected.get("libsignal/throughput"),
        "row_count": len(rows),
    }
    summary.update(selected)
    summary["phase5/metric_source"] = "libsignal_dtl_log"

    summary_path = Path(args.summary_path).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
