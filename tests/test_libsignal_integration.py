import importlib.util
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_MODULE_PATH = Path(__file__).resolve().parents[1] / "sumo_rl" / "integrations" / "libsignal.py"
_SPEC = importlib.util.spec_from_file_location("libsignal_helper", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_phase5_trace_row = _MODULE.build_phase5_trace_row
parse_libsignal_dtl_log = _MODULE.parse_libsignal_dtl_log
select_libsignal_trace_row = _MODULE.select_libsignal_trace_row


def test_parse_and_build_phase5_row() -> None:
    scratch_dir = Path(__file__).resolve().parents[1] / "outputs" / "_test_libsignal"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    log_path = scratch_dir / "sample_DTL.log"
    log_path.write_text(
        "\n".join(
            [
                "dqn\tTRAIN\t10\t101.5\t2.5\t-1.0\t3.0\t1.2\t8",
                "dqn\tTEST\t10\t99.5\t2.0\t-0.5\t2.0\t1.0\t9",
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_libsignal_dtl_log(log_path)
    assert len(rows) == 2

    selected = select_libsignal_trace_row(rows)
    assert selected["libsignal/mode"] == "TEST"
    assert selected["libsignal/travel_time"] == 99.5

    row = build_phase5_trace_row(selected, run_idx=1, run_seed=7, experiment_name="libsignal_idqn_resco_grid4x4")
    assert row["backend"] == "libsignal"
    assert row["phase5/model"] == "dqn"
    assert row["resco_trip_time"] == 99.5
    assert row["resco_avg_delay"] == 1.0
    assert row["episode/reward"] == -0.5
