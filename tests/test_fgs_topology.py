from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.agents.fgs.topology import extract_tls_topology


def test_fgs_topology_uses_controlled_junction_for_mismatched_tls_id(tmp_path):
    net_file = tmp_path / "mismatched_tls.net.xml"
    net_file.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
    <edge id="in_a" from="source" to="cluster_a" priority="1">
        <lane id="in_a_0" index="0" speed="10.00" length="30.00" shape="-30.00,0.00 0.00,0.00"/>
    </edge>
    <edge id="ab" from="cluster_a" to="mid" priority="1">
        <lane id="ab_0" index="0" speed="10.00" length="50.00" shape="0.00,0.00 50.00,0.00"/>
    </edge>
    <edge id="bc" from="mid" to="tls_b" priority="1">
        <lane id="bc_0" index="0" speed="10.00" length="50.00" shape="50.00,0.00 100.00,0.00"/>
    </edge>
    <connection from="in_a" to="ab" fromLane="0" toLane="0" tl="program_a" linkIndex="0" dir="s" state="O"/>
    <connection from="ab" to="bc" fromLane="0" toLane="0"/>
    <tlLogic id="program_a" type="static" programID="0" offset="0"><phase duration="30" state="G"/></tlLogic>
    <tlLogic id="tls_b" type="static" programID="0" offset="0"><phase duration="30" state="G"/></tlLogic>
    <junction id="source" type="dead_end" x="-30.00" y="0.00" incLanes="" intLanes=""/>
    <junction id="cluster_a" type="traffic_light" x="0.00" y="0.00" incLanes="in_a_0" intLanes=""/>
    <junction id="mid" type="priority" x="50.00" y="0.00" incLanes="ab_0" intLanes=""/>
    <junction id="tls_b" type="traffic_light" x="100.00" y="0.00" incLanes="bc_0" intLanes=""/>
</net>
""",
        encoding="utf-8",
    )

    topology = extract_tls_topology(net_file)

    assert topology.workers == ["program_a", "tls_b"]
    assert topology.positions["program_a"] == (0.0, 0.0)
    assert topology.directed_edges == [("program_a", "tls_b")]
    assert topology.super_edges[0].path_node_ids == ["cluster_a", "mid", "tls_b"]


def test_fgs_topology_keeps_all_ingolstadt21_tls_programs():
    net_file = ROOT / "sumo_rl/nets/RESCO/ingolstadt21/ingolstadt21.net.xml"

    topology = extract_tls_topology(net_file)

    assert len(topology.workers) == 21
    assert {
        "gneJ143",
        "gneJ207",
        "gneJ208",
        "gneJ210",
        "gneJ255",
        "gneJ257",
    }.issubset(topology.workers)
