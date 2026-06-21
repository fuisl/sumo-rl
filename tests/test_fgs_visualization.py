from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.render_fgs_topology import render_fgs_visualization


def _write_tiny_mismatched_net(path: Path) -> None:
    path.write_text(
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


def test_fgs_visualization_writes_pipeline_svgs_and_json(tmp_path):
    net_file = tmp_path / "tiny.net.xml"
    _write_tiny_mismatched_net(net_file)

    paths = render_fgs_visualization(net_file, tmp_path / "visualization", width=700)

    for path in paths.values():
        assert path.exists()

    node_svg = paths["node_svg"].read_text(encoding="utf-8")
    topology_svg = paths["topology_svg"].read_text(encoding="utf-8")
    node_json = json.loads(paths["node_json"].read_text(encoding="utf-8"))
    topology_json = json.loads(paths["topology_json"].read_text(encoding="utf-8"))

    assert 'id="underlying-network"' in node_svg
    assert 'class="ordinary-junction"' in node_svg
    assert 'class="tls-program-controlled"' in node_svg
    assert "program_a" in node_svg
    assert 'id="fgs-super-edges"' in topology_svg
    assert 'marker-end="url(#fgs-arrow)"' in topology_svg
    assert "program_a" in topology_svg
    assert node_json["counts"]["junctions"] == 4
    assert node_json["tls_program_to_junction"]["program_a"] == "cluster_a"
    assert topology_json["workers"] == ["program_a", "tls_b"]


def test_fgs_visualization_ingolstadt21_metadata_keeps_all_tls_workers(tmp_path):
    net_file = ROOT / "sumo_rl/nets/RESCO/ingolstadt21/ingolstadt21.net.xml"

    paths = render_fgs_visualization(net_file, tmp_path / "ingolstadt21", width=900)

    topology_json = json.loads(paths["topology_json"].read_text(encoding="utf-8"))
    assert len(topology_json["workers"]) == 21
    assert {
        "gneJ143",
        "gneJ207",
        "gneJ208",
        "gneJ210",
        "gneJ255",
        "gneJ257",
    }.issubset(topology_json["workers"])
