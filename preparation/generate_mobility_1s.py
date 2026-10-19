#!/usr/bin/env python3
"""Sample existing straight 10 s simulation segments at 1 s; no API calls.

Keeps the path and stop schedule, recalculates geometric channels, and applies
the same static-channel offsets as the current manuscript's dynamic input.
All generated data files are new; existing output files are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
EXPERIMENT = HERE
sys.path.insert(0, str(EXPERIMENT))
import generate_scenarios as geometry
from generate_mobility import LONG_FIELDS, USER_SUMMARY_FIELDS, calculate_channel, write_csv
from continuation.generate_anchored_channel import DIAGNOSTICS, SIGNAL_FIELDS, calculate_cqi

SCENES = geometry.SCENES
XYZ = ("X", "Y", "Z")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def grouped(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["RX_ID"]].append(row)
    for history in result.values():
        history.sort(key=lambda r: int(r["Time_s"]))
        assert [int(r["Time_s"]) for r in history] == list(range(0, 301, 10))
    assert set(result) == {str(i) for i in range(1, 31)}
    return result


def summarize(scene, uid, rows):
    states = Counter(r["Movement_State"] for r in rows[1:])
    cqis = [int(r["CQI"]) for r in rows]
    return dict(Scenario=scene, RX_ID=uid, Mobility_Profile=rows[0]["Mobility_Profile"],
                Total_Distance_m=rows[-1]["Cumulative_Distance_m"],
                Moving_Intervals=states["moving"], Paused_Intervals=states["paused"],
                Blocked_Intervals=states["blocked"], CQI_Initial=cqis[0], CQI_Final=cqis[-1],
                CQI_Min=min(cqis), CQI_Max=max(cqis),
                CQI_Changes=sum(a != b for a, b in zip(cqis, cqis[1:])))


def build_scene(scene, raw_path, anchored_path, model, output):
    raw_old, anchored_old = grouped(read_csv(raw_path)), grouped(read_csv(anchored_path))
    buildings, tx, _, _ = geometry.scene_geometry(model, scene)
    raw_rows, rows, summaries, comparisons = [], [], [], []
    max_distance_error = 0.0
    for uid in sorted(raw_old, key=int):
        history, old_anchor = raw_old[uid], anchored_old[uid]
        start = old_anchor[0]
        raw_start = history[0]
        for r, a in zip(history, old_anchor):
            assert all(r[k] == a[k] for k in LONG_FIELDS if k not in SIGNAL_FIELDS)
        snr_offset = Decimal(start["SNR_dB"]) - Decimal(raw_start["SNR_dB"])
        rx_offset = Decimal(start["RX_Power_dBm"]) - Decimal(raw_start["RX_Power_dBm"])
        user_rows, total, previous_xyz = [], 0.0, None
        for t in range(301):
            left_index = t // 10
            right_index = min(30, (t + 9) // 10)
            left, right = history[left_index], history[right_index]
            fraction = Decimal(t % 10) / Decimal(10)
            # Decimal interpolation followed by the source's micrometre export
            # precision preserves every existing 10 s coordinate verbatim.
            xyz_text = {k: (left[k] if t % 10 == 0 else
                           f'{Decimal(left[k]) + fraction * (Decimal(right[k]) - Decimal(left[k])):.6f}')
                        for k in XYZ}
            xyz = tuple(float(xyz_text[k]) for k in XYZ)
            step = math.dist(xyz, previous_xyz) if previous_xyz is not None else 0.0
            total += step
            previous_xyz = xyz
            channel = calculate_channel(model, tx, xyz, buildings)
            if t % 10 == 0:
                assert channel == {k: left[k] for k in SIGNAL_FIELDS}, (scene, uid, t, "raw channel")
            raw = {
                "Scenario": scene, "Frame_Index": t, "Time_s": t, "RX_ID": uid,
                **xyz_text, **channel,
                "User_Request": start["User_Request"], "Request_Label": start["Request_Label"],
                "Mobility_Profile": start["Mobility_Profile"],
                "Movement_State": "initial" if t == 0 else history[right_index]["Movement_State"],
                "Step_Distance_m": f"{step:.6f}", "Speed_mps": f"{step:.6f}",
                "Cumulative_Distance_m": f"{total:.6f}",
            }
            row = dict(raw)
            row.update({
                "SNR_dB": f'{Decimal(channel["SNR_dB"]) + snr_offset:.6f}',
                "RX_Power_dBm": f'{Decimal(channel["RX_Power_dBm"]) + rx_offset:.6f}',
                "raw_model_snr_db": channel["SNR_dB"],
                "raw_model_rx_power_dbm": channel["RX_Power_dBm"],
                "raw_model_cqi": channel["CQI"], "raw_model_los": channel["LOS"],
                "channel_snr_offset_db": str(snr_offset), "channel_rx_offset_db": str(rx_offset),
                "legacy_static_snr_db": start["legacy_static_snr_db"],
                "legacy_static_rx_power_dbm": start["legacy_static_rx_power_dbm"],
                "legacy_static_cqi": start["legacy_static_cqi"],
                "legacy_static_los": start["legacy_static_los"],
                "exported_los_origin": "current_geometry",
            })
            row["CQI"] = str(calculate_cqi(row["SNR_dB"]))
            if t == 0 or row["Mobility_Profile"] == "static":
                row.update({k: start[k] for k in SIGNAL_FIELDS})
                row["exported_los_origin"] = "legacy_static_snapshot"
            if t % 10 == 0:
                old = old_anchor[left_index]
                assert all(row[k] == old[k] for k in (*XYZ, *SIGNAL_FIELDS, *DIAGNOSTICS))
                error = abs(float(row["Cumulative_Distance_m"]) - float(old["Cumulative_Distance_m"]))
                assert error <= 2e-6, (scene, uid, t, error)
                max_distance_error = max(max_distance_error, error)
                comparisons.append({"Scenario": scene, "RX_ID": uid, "Time_s": t,
                                    "Source_Frame_Index": left_index, "New_Frame_Index": t,
                                    "XYZ_Exact_Match": True, "Signal_Exact_Match": True,
                                    "Diagnostics_Exact_Match": True,
                                    "Cumulative_Distance_Difference_m": f"{error:.9f}"})
            raw_rows.append(raw)
            rows.append(row)
            user_rows.append(row)
        summaries.append(summarize(scene, uid, user_rows))
    order = lambda r: (int(r["Time_s"]), int(r["RX_ID"]))
    rows.sort(key=order)
    raw_rows.sort(key=order)
    write_csv(output / f"{scene}.csv", [*LONG_FIELDS, *DIAGNOSTICS], rows)
    by_time = defaultdict(list)
    for row in rows:
        by_time[row["Time_s"]].append(row)
    for t, snapshot in by_time.items():
        fields = [*geometry.FIELDS, *DIAGNOSTICS]
        write_csv(output / "frames" / f"frame_{t:03d}" / f"{scene}.csv", fields,
                  [{k: row[k] for k in fields} for row in snapshot])
    report = {
        "users": 30, "frames": 301, "rows": len(rows),
        "profile_counts": dict(Counter(r["Mobility_Profile"] for r in rows[:30])),
        "original_anchor_records": len(comparisons), "added_position_samples": 8100,
        "all_10s_coordinates_channels_and_diagnostics_match_exactly": True,
        "max_10s_cumulative_distance_difference_m": max_distance_error,
        "users_with_cqi_changes": sum(s["CQI_Changes"] > 0 for s in summaries),
        "cqi_change_events": sum(s["CQI_Changes"] for s in summaries),
        "blocked_intervals": sum(s["Blocked_Intervals"] for s in summaries),
        "cqi_range": [min(int(r["CQI"]) for r in rows), max(int(r["CQI"]) for r in rows)],
    }
    assert report["profile_counts"] == {"static": 10, "continuous": 10, "intermittent": 10}
    print(f"{scene}: {len(rows)} rows, original anchors preserved", flush=True)
    return rows, raw_rows, summaries, comparisons, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=EXPERIMENT / "data/mobility_5min_from_static_20261014")
    parser.add_argument("--anchored-input", type=Path, default=EXPERIMENT / "data/mobility_5min_static_terminal_20261014")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "outputs/mobility_regenerated")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / n).exists() for n in ("generation.json", "center.csv", "frames", "dynamic_users_1s.csv")):
        parser.error("Data already exist at output. Use a new --output; no overwrite is allowed.")
    src_meta = json.loads((args.anchored_input / "generation.json").read_text())
    input_paths = [args.raw_input / "generation.json", args.anchored_input / "generation.json"]
    for scene in SCENES:
        input_paths.extend([args.raw_input / f"{scene}.csv", args.anchored_input / f"{scene}.csv",
                            args.anchored_input / "initial_static" / f"{scene}.csv",
                            EXPERIMENT / src_meta["scenarios"][scene]["map_file"]])
    input_paths.extend(args.anchored_input / n for n in ("user_profiles.csv", "static_initial_comparison.csv"))
    source_hashes = {str(p.resolve()): sha(p) for p in input_paths}
    for scene in SCENES:
        meta = src_meta["scenarios"][scene]
        assert sha(EXPERIMENT / meta["map_file"]) == meta["map_sha256"]
    model = geometry.load_model()
    with tempfile.TemporaryDirectory(prefix=".mobility_1s_", dir=output.parent) as tmp:
        stage = Path(tmp)
        all_rows, all_raw, all_summaries, all_comparisons, reports = [], [], [], [], {}
        for scene in SCENES:
            rows, raw, summaries, comparisons, report = build_scene(
                scene, args.raw_input / f"{scene}.csv", args.anchored_input / f"{scene}.csv", model, stage)
            all_rows.extend(rows)
            all_raw.extend(raw)
            all_summaries.extend(summaries)
            all_comparisons.extend(comparisons)
            reports[scene] = report
        write_csv(stage / "dynamic_users_1s.csv", [*LONG_FIELDS, *DIAGNOSTICS], all_rows)
        write_csv(stage / "dynamic_users_raw_model_1s.csv", LONG_FIELDS, all_raw)
        write_csv(stage / "user_summary.csv", USER_SUMMARY_FIELDS, all_summaries)
        write_csv(stage / "anchor_comparison_10s.csv", list(all_comparisons[0]), all_comparisons)
        timeline = [{"Frame_Index": t, "Time_s": t, "Time_Label": f"{t//60:02d}:{t%60:02d}",
                     "Frame_Directory": f"frames/frame_{t:03d}",
                     **{f"{s}_csv": f"frames/frame_{t:03d}/{s}.csv" for s in SCENES},
                     "Position_Origin": "original_10s_anchor" if t % 10 == 0 else "segment_resampling",
                     "Source_Left_Time_s": (t // 10) * 10,
                     "Source_Right_Time_s": min(300, ((t + 9) // 10) * 10),
                     "Interpolation_Fraction": f"{(t % 10) / 10:.1f}"}
                    for t in range(301)]
        write_csv(stage / "timeline.csv", list(timeline[0]), timeline)
        for name in ("user_profiles.csv", "static_initial_comparison.csv"):
            shutil.copyfile(args.anchored_input / name, stage / name)
        shutil.copytree(args.anchored_input / "initial_static", stage / "initial_static")
        for label, source_dir in (("raw_10s", args.raw_input), ("anchored_10s", args.anchored_input)):
            shutil.copyfile(source_dir / "generation.json", stage / f"source_generation_{label}.json")
        validation = {"passed": True, "total_users": 90, "total_rows": len(all_rows),
                      "frames_per_scene": 301, "sample_interval_s": 1,
                      "original_anchor_records": 2790, "added_position_samples": 24300,
                      "scenarios": reports, "source_files_unchanged": True, "api_calls": 0,
                      "scope": "Generation checks. Independent geometric/channel audit is separate."}
        write_json(stage / "validation.json", validation)
        keep = ("base_seed", "scene_seed_rule", "static_selection", "moving_fraction",
                "intermittent_fraction_of_movers", "nominal_speed_range_mps", "profile_assignment",
                "bounds_policy", "channel_model", "channel_parameters", "user_identity")
        metadata = {k: deepcopy(src_meta[k]) for k in keep}
        metadata.update({
            "scenario_variant": "existing_10s_paths_resampled_at_1s_static_terminal_channels",
            "dataset_date": "2026-10-15", "duration_s": 300, "interval_s": 1, "frame_count": 301,
            "main_csv": "dynamic_users_1s.csv", "raw_model_csv": "dynamic_users_raw_model_1s.csv",
            "time_origin": "t=0 preserves the original static positions, requests and channel values.",
            "interval_semantics": "Frame t is an instantaneous sample at integer second t; movement fields describe (t-1,t]. Frame 0 has zero distance/speed. This is not a one-second averaged network measurement.",
            "movement_model": "Original 10 s correlated-random-walk segments, each subdivided into 10 straight constant-speed one-second segments. No new RNG draws or changed pause windows.",
            "sampling_method": "Piecewise linear XYZ interpolation on the exported original 10 s path, rounded to 6 decimals. Exact original anchors retained. Step and cumulative distances recomputed from delivered coordinates.",
            "data_provenance": "Derived simulation input; added samples are not independently observed or measured data.",
            "coordinate_and_signal_decimals": 6,
            "channel_anchor": {**src_meta["channel_anchor"], "scope": "All 90 users, all 301 samples.",
                               "signal_precision": "Geometric LOS/path loss evaluated at each exported position; raw signal to 6 decimals, then Decimal per-user static offsets; no temporal channel interpolation."},
            "source_raw_directory": str(args.raw_input.resolve()),
            "source_anchored_directory": str(args.anchored_input.resolve()),
            "source_sha256": source_hashes,
            "code_sha256": {str(p.resolve()): sha(p) for p in (
                Path(__file__), EXPERIMENT / "generate_mobility.py", EXPERIMENT / "generate_scenarios.py",
                EXPERIMENT / "continuation/generate_anchored_channel.py",
                EXPERIMENT / "vendor/upstream/RayTracing_cqi.py")},
            "software": {"python": platform.python_version(), **{
                n: importlib.metadata.version(n) for n in ("numpy", "matplotlib", "pyproj", "shapely")}},
            "scenarios": {s: {**deepcopy(src_meta["scenarios"][s]), "statistics": reports[s]} for s in SCENES},
            "limitations": [
                "Higher sampling density preserves existing paths; it does not add independent motion realism or measured high-frequency fading.",
                "Original 10 s heading changes and movement/pause schedules are retained.",
                "Original static channel offsets are retained, including north source/model discrepancies.",
                "Four source users inside building footprints remain stationary; roads and shoreline walkability are not modeled.",
                "These files contain mobility and modeled channels, not newly run allocation, throughput, fairness, or control outcomes.",
                "Existing runners contain 10 s / 31-frame assumptions and need a separate scheduling update before using this input for completion-triggered control.",
            ],
            "output_sha256": {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob("*")) if p.is_file()},
        })
        assert all(sha(Path(p)) == digest for p, digest in source_hashes.items()), "Source changed during generation"
        write_json(stage / "generation.json", metadata)
        for item in stage.iterdir():
            if (output / item.name).exists():
                raise FileExistsError(output / item.name)
        for item in stage.iterdir():
            shutil.move(str(item), output / item.name)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
