from __future__ import annotations

from pathlib import Path

import pytest

from sumo_rl.experiments.validation_metrics import (
    build_all_demand_metrics,
    build_completed_trip_metrics,
    gini_coefficient,
    jain_index,
    load_statistic_output,
)
from sumo_rl.util.statistics_output import parse_statistic_output


pytestmark = pytest.mark.core_fast


def test_parse_statistic_output_reads_vehicle_and_trip_sections(tmp_path: Path):
    statistic_xml = tmp_path / "stats.xml"
    statistic_xml.write_text(
        """<statistics>
  <performance clockBegin="1" clockEnd="2" clockDuration="3" traciDuration="0.5" realTimeFactor="4.0" vehicleUpdatesPerSecond="2000" />
  <timing begin="0" end="3600" duration="3600" />
  <vehicles loaded="100" inserted="90" running="10" waiting="5" />
  <teleports total="7" jam="4" yield="2" wrongLane="1" />
  <safety collisions="3" emergencyStops="6" />
  <vehicleTripStatistics count="80" speed="9.5" duration="120" waitingTime="50" timeLoss="40" departDelay="5" departDelayWaiting="30" totalTravelTime="9600" totalDepartDelay="450" />
</statistics>""",
        encoding="utf-8",
    )

    parsed = parse_statistic_output(statistic_xml)

    assert parsed.vehicles_loaded == 100.0
    assert parsed.vehicles_inserted == 90.0
    assert parsed.teleports_jam == 4.0
    assert parsed.trip_speed == 9.5
    assert parsed.trip_total_travel_time == 9600.0


def test_load_statistic_output_maps_public_metric_names(tmp_path: Path):
    statistic_xml = tmp_path / "stats.xml"
    statistic_xml.write_text(
        """<statistics>
  <performance clockDuration="12" traciDuration="1.5" realTimeFactor="6.0" vehicleUpdatesPerSecond="2500" />
  <vehicles loaded="40" inserted="35" running="5" waiting="2" />
  <vehicleTripStatistics speed="8.0" duration="100" waitingTime="30" timeLoss="20" departDelay="4" departDelayWaiting="10" totalTravelTime="3500" totalDepartDelay="140" />
</statistics>""",
        encoding="utf-8",
    )

    loaded = load_statistic_output(statistic_xml)

    assert loaded["sumo_clock_duration"] == 12.0
    assert loaded["loaded_vehicle_count"] == 40.0
    assert loaded["avg_speed_completed"] == 8.0
    assert loaded["total_depart_delay_statistics"] == 140.0


def test_build_completed_trip_metrics_keeps_existing_completed_trip_values():
    seed_row = {
        "tripinfo/finished_count": 20.0,
        "final/resco/avg_delay": 11.5,
        "final/resco/avg_delay_std": 1.2,
        "validation/resco_delay_max": 18.0,
        "final/resco/trip_time": 55.0,
        "tripinfo/std_duration": 4.0,
        "final/resco/wait": 8.5,
        "final/resco/wait_std": 2.1,
        "validation/resco_wait_max": 20.0,
        "tripinfo/avg_time_loss": 7.0,
        "tripinfo/std_time_loss": 1.7,
    }

    metrics = build_completed_trip_metrics(seed_row)

    assert metrics["completed_trip_count"] == 20.0
    assert metrics["avg_delay_completed"] == 11.5
    assert metrics["std_delay_completed"] == 1.2
    assert metrics["avg_trip_time_completed"] == 55.0
    assert metrics["avg_waiting_time_completed"] == 8.5


def test_build_all_demand_metrics_uses_loaded_denominator_when_available():
    seed_row = {"final/efficiency/total_arrived": 80.0}
    stats = {
        "loaded_vehicle_count": 100.0,
        "inserted_vehicle_count": 90.0,
        "running_vehicle_count": 10.0,
        "undeparted_vehicle_count": 5.0,
        "avg_trip_time_statistics": 120.0,
        "avg_time_loss_statistics": 40.0,
        "avg_depart_delay_statistics": 5.0,
        "avg_depart_delay_waiting_statistics": 30.0,
        "total_travel_time_statistics": 10800.0,
        "total_depart_delay_statistics": 450.0,
    }

    metrics = build_all_demand_metrics(seed_row, stats)

    assert metrics["completion_ratio"] == 0.8
    assert metrics["completion_ratio_per_inserted"] == 80.0 / 90.0
    assert metrics["total_depart_delay_all"] == 600.0
    assert metrics["observed_total_travel_time_and_delay"] == 11400.0
    assert metrics["avg_observed_system_time_all"] == 114.0
    assert metrics["avg_observed_delay_all"] == 42.0


def test_fairness_indices_handle_zero_and_skewed_inputs():
    assert jain_index([0.0, 0.0, 0.0]) == 1.0
    assert gini_coefficient([0.0, 0.0, 0.0]) == 0.0
    assert jain_index([1.0, 1.0, 1.0]) == 1.0
    assert gini_coefficient([0.0, 1.0, 3.0]) > 0.0
