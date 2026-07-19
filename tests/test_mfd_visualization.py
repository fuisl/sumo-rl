from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.plot_mfd import build_mfd_rows


def test_build_mfd_rows_uses_lane_km_density_and_speed_flow(tmp_path):
    net_file = tmp_path / "toy.net.xml"
    net_file.write_text(
        """<net>
  <edge id="edge_a">
    <lane id="edge_a_0" length="100.0"/>
  </edge>
  <edge id="edge_b">
    <lane id="edge_b_0" length="200.0"/>
    <lane id="edge_b_1" length="200.0"/>
  </edge>
  <edge id=":internal" function="internal">
    <lane id=":internal_0" length="999.0"/>
  </edge>
</net>
""",
        encoding="utf-8",
    )
    trace_path = tmp_path / "resco_toy__fixed_time" / "trip_trace.json"
    trace_path.parent.mkdir()
    trace_path.write_text(
        json.dumps(
            {
                "metadata": {"scenario": "resco_toy", "algorithm_kind": "fixed_time", "seed": 7},
                "network": {"net_file": str(net_file), "road_polylines": []},
                "frames": [
                    {
                        "time": 10.0,
                        "vehicles": [
                            {"id": "veh_1", "speed": 10.0},
                            {"id": "veh_2", "speed": 20.0},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = build_mfd_rows([trace_path])

    assert len(rows) == 1
    row = rows[0]
    assert row["scenario"] == "resco_toy"
    assert row["control"] == "fixed_time"
    assert row["lane_count"] == 3
    assert math.isclose(row["lane_length_km"], 0.5)
    assert math.isclose(row["lane_density_veh_per_km"], 4.0)
    assert math.isclose(row["mean_speed_kmh"], 54.0)
    assert math.isclose(row["lane_flow_veh_per_hour"], 216.0)
    assert math.isclose(row["production_veh_km_per_hour"], 108.0)
