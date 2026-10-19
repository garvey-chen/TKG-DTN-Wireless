#!/usr/bin/env python3
"""Preserve the mobility paths and anchor every user's channel to static t=0.

Creates a new dataset only. Uses the existing deterministic channel increments,
never new fading, positions, admissions, allocations, or model/API calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
from zoneinfo import ZoneInfo


SCENES = ("center", "north", "south")
SIGNAL_FIELDS = ("SNR_dB", "RX_Power_dBm", "CQI", "LOS")
SNAPSHOT_FIELDS = ("RX_ID", "X", "Y", "Z", *SIGNAL_FIELDS, "User_Request", "Request_Label")
DIAGNOSTICS = ("raw_model_snr_db", "raw_model_rx_power_dbm", "raw_model_cqi", "raw_model_los",
               "channel_snr_offset_db", "channel_rx_offset_db", "legacy_static_snr_db",
               "legacy_static_rx_power_dbm", "legacy_static_cqi", "legacy_static_los", "exported_los_origin")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows: list[dict], fields=None) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def calculate_cqi(snr: str | Decimal) -> int:
    # Exact arithmetic order of vendor/upstream/RayTracing_cqi.py::calculate_cqi.
    clipped = max(-10.0, min(30.0, float(snr)))
    normalized = (clipped - -10.0) / (30.0 - -10.0)
    return round(1 + normalized * 14)


def anchor_scene(scene: str, input_dir: Path, static_dir: Path) -> tuple[list[dict], list[dict], dict, list[dict]]:
    original_rows = read_rows(static_dir / f"{scene}.csv")[:30]
    legacy = {row["RX_ID"]: row for row in original_rows}
    raw = read_rows(input_dir / f"{scene}.csv")
    zero = {row["RX_ID"]: row for row in raw if row["Frame_Index"] == "0"}
    if set(zero) != set(legacy) or len(raw) != 930:
        raise ValueError(f"Unexpected population/timeline: {scene}")
    rows, offsets = [], []
    for uid, initial in zero.items():
        old = legacy[uid]
        if calculate_cqi(old["SNR_dB"]) != int(old["CQI"]):
            raise ValueError(f"Legacy SNR/CQI is not self-consistent: {scene}/{uid}")
        if any(Decimal(initial[key]) != Decimal(old[key]) for key in ("X", "Y", "Z")):
            raise ValueError(f"Initial position differs: {scene}/{uid}")
        offsets.append({"Scenario": scene, "RX_ID": uid,
                        "Static_SNR_dB": old["SNR_dB"], "Raw_Model_Initial_SNR_dB": initial["SNR_dB"],
                        "SNR_Offset_dB": str(Decimal(old["SNR_dB"]) - Decimal(initial["SNR_dB"])),
                        "Static_RX_Power_dBm": old["RX_Power_dBm"],
                        "Raw_Model_Initial_RX_Power_dBm": initial["RX_Power_dBm"],
                        "RX_Offset_dB": str(Decimal(old["RX_Power_dBm"]) - Decimal(initial["RX_Power_dBm"])),
                        "Static_CQI": old["CQI"], "Raw_Model_Initial_CQI": initial["CQI"],
                        "Anchored_Initial_CQI": old["CQI"], "Static_LOS": old["LOS"],
                        "Raw_Model_Initial_LOS": initial["LOS"], "Mobility_Profile": initial["Mobility_Profile"]})
    for current in raw:
        uid = current["RX_ID"]
        old, initial = legacy[uid], zero[uid]
        row = dict(current)
        snr_offset = Decimal(old["SNR_dB"]) - Decimal(initial["SNR_dB"])
        rx_offset = Decimal(old["RX_Power_dBm"]) - Decimal(initial["RX_Power_dBm"])
        row.update({
            "SNR_dB": f'{Decimal(current["SNR_dB"]) + snr_offset:.6f}',
            "RX_Power_dBm": f'{Decimal(current["RX_Power_dBm"]) + rx_offset:.6f}',
            "raw_model_snr_db": current["SNR_dB"], "raw_model_rx_power_dbm": current["RX_Power_dBm"],
            "raw_model_cqi": current["CQI"], "raw_model_los": current["LOS"],
            "channel_snr_offset_db": str(snr_offset), "channel_rx_offset_db": str(rx_offset),
            "legacy_static_snr_db": old["SNR_dB"], "legacy_static_rx_power_dbm": old["RX_Power_dBm"],
            "legacy_static_cqi": old["CQI"], "legacy_static_los": old["LOS"],
            "exported_los_origin": "current_geometry",
        })
        row["CQI"] = str(calculate_cqi(row["SNR_dB"]))
        if current["Frame_Index"] == "0" or current["Mobility_Profile"] == "static":
            # Preserve all source channel values exactly for t=0 and stationary
            # receivers; raw geometry LOS remains separately visible.
            row.update({key: old[key] for key in SIGNAL_FIELDS})
            row["exported_los_origin"] = "legacy_static_snapshot"
        rows.append(row)

    summaries = []
    for uid in legacy:
        history = [row for row in rows if row["RX_ID"] == uid]
        cqis = [int(row["CQI"]) for row in history]
        states = Counter(row["Movement_State"] for row in history[1:])
        summaries.append({"Scenario": scene, "RX_ID": uid, "Mobility_Profile": history[0]["Mobility_Profile"],
                          "Total_Distance_m": history[-1]["Cumulative_Distance_m"],
                          "Moving_Intervals": states["moving"], "Paused_Intervals": states["paused"],
                          "Blocked_Intervals": states["blocked"], "CQI_Initial": cqis[0], "CQI_Final": cqis[-1],
                          "CQI_Min": min(cqis), "CQI_Max": max(cqis),
                          "CQI_Changes": sum(left != right for left, right in zip(cqis, cqis[1:]))})

    # Independent invariants on the exact records to be serialized.
    assert len(rows) == 930
    preserved_fields = [key for key in raw[0] if key not in SIGNAL_FIELDS]
    for old_row, new_row in zip(raw, rows):
        uid, frame = new_row["RX_ID"], int(new_row["Frame_Index"])
        source_row, start = legacy[uid], zero[uid]
        assert all(old_row[key] == new_row[key] for key in preserved_fields)
        assert new_row["User_Request"] == source_row["User_Request"]
        assert new_row["Request_Label"] == source_row["Request_Label"]
        assert Decimal(new_row["SNR_dB"]) == Decimal(source_row["SNR_dB"]) + Decimal(old_row["SNR_dB"]) - Decimal(start["SNR_dB"])
        assert Decimal(new_row["RX_Power_dBm"]) == Decimal(source_row["RX_Power_dBm"]) + Decimal(old_row["RX_Power_dBm"]) - Decimal(start["RX_Power_dBm"])
        assert calculate_cqi(new_row["SNR_dB"]) == int(new_row["CQI"])
        if frame == 0 or new_row["Mobility_Profile"] == "static":
            assert all(new_row[key] == source_row[key] for key in SIGNAL_FIELDS)
    for uid in legacy:
        history = [row for row in rows if row["RX_ID"] == uid]
        assert [int(row["Time_s"]) for row in history] == list(range(0, 301, 10))
        for previous, current in zip(history, history[1:]):
            if all(previous[key] == current[key] for key in ("X", "Y", "Z")):
                assert all(Decimal(previous[key]) == Decimal(current[key]) for key in SIGNAL_FIELDS)
    report = {
        "users": 30, "rows": len(rows), "frames": 31,
        "profile_counts": dict(Counter(row["Mobility_Profile"] for row in zero.values())),
        "exact_source_t0_channel_matches": 30,
        "stationary_users_with_unchanged_legacy_channel": sum(row["Mobility_Profile"] == "static" for row in zero.values()),
        "identical_original_trajectory_and_requests": True,
        "source_snr_cqi_self_consistent": True,
        "anchored_formula_verified_all_rows": True,
        "raw_model_frame0_cqi_mismatch_ids": [uid for uid in legacy if legacy[uid]["CQI"] != zero[uid]["CQI"]],
        "raw_model_frame0_los_mismatch_ids": [uid for uid in legacy if legacy[uid]["LOS"] != zero[uid]["LOS"]],
        "source_vs_model_snr_max_abs_difference_db": max(abs(float(row["SNR_Offset_dB"])) for row in offsets),
        "users_with_cqi_changes": sum(row["CQI_Changes"] > 0 for row in summaries),
        "cqi_change_events": sum(row["CQI_Changes"] for row in summaries),
        "paused_channels_unchanged": True,
    }
    assert report["profile_counts"] == {"static": 10, "continuous": 10, "intermittent": 10}
    return rows, summaries, report, offsets


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    workspace = next(parent for parent in project.parents if (parent / "论文大修").is_dir())
    package = workspace / "论文大修/05_无线实验材料/02_WirelessAgent实验/无线实验重写稿_20261012_五次重复双栏/静态衔接动态场景_20261014/动态实验_30用户_静态终态延续_20261014"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=project / "data/mobility_5min_from_static_20261014")
    parser.add_argument("--static-dir", type=Path, default=project.parent / "experiments_Wireless_v2/data/scenarios")
    parser.add_argument("--output", type=Path, default=project / "data/mobility_5min_static_terminal_20261014")
    parser.add_argument("--package-dir", type=Path, default=package)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    package_csv = args.package_dir / "dynamic_users_static_terminal.csv"
    package_note = args.package_dir / "信道衔接说明.txt"
    if package_csv.exists() or package_note.exists():
        raise FileExistsError("Refusing to overwrite existing package channel artifacts")
    original_generation = json.loads((args.input_dir / "generation.json").read_text(encoding="utf-8"))
    outputs, reports, all_rows, summaries, offsets = {}, {}, [], [], []
    for scene in SCENES:
        rows, scene_summaries, report, scene_offsets = anchor_scene(scene, args.input_dir, args.static_dir)
        reports[scene] = report
        all_rows.extend(rows)
        summaries.extend(scene_summaries)
        offsets.extend(scene_offsets)
        outputs[f"{scene}.csv"] = csv_bytes(rows)
        outputs[f"initial_static/{scene}.csv"] = csv_bytes(read_rows(args.static_dir / f"{scene}.csv")[:30])
        for frame in range(31):
            snapshot = [row for row in rows if int(row["Frame_Index"]) == frame]
            outputs[f"frames/frame_{frame:03d}/{scene}.csv"] = csv_bytes(snapshot, (*SNAPSHOT_FIELDS, *DIAGNOSTICS))
    outputs["dynamic_users_from_static.csv"] = csv_bytes(all_rows)
    outputs["user_summary.csv"] = csv_bytes(summaries)
    outputs["static_initial_comparison.csv"] = csv_bytes(offsets)
    for name in ("timeline.csv", "user_profiles.csv"):
        outputs[name] = (args.input_dir / name).read_bytes()
    outputs["raw_model_initial_comparison.csv"] = (args.input_dir / "static_initial_comparison.csv").read_bytes()
    outputs["source_generation.json"] = (args.input_dir / "generation.json").read_bytes()
    outputs["source_validation.json"] = (args.input_dir / "validation.json").read_bytes()
    validation = {"passed": True, "total_users": 90, "total_rows": 2790, "scenarios": reports,
                  "source_trajectories_unchanged": True, "api_calls": 0,
                  "all_initial_channel_fields_match_static": True,
                  "all_30_stationary_users_preserve_static_channel_for_300s": True}
    outputs["validation.json"] = json_bytes(validation)

    # The supplied north source, map and code have identical upstream copies.
    # These hashes support provenance, not a guess at historical model settings.
    upstream = workspace / "无线场景补充实验/WirelessAgent_R1-main/WirelessAgent_R1-main"
    diagnostic_paths = {
        "static_north_csv": args.static_dir / "north.csv",
        "original_upstream_north_csv": upstream / "Knowledge_Base/ray_tracing_results_north.csv",
        "current_north_map": project / "data/maps/HKUST_North.osm",
        "original_upstream_north_map": upstream / "Knowledge_Base/HKUST_North.osm",
        "current_channel_code": project / "vendor/upstream/RayTracing_cqi.py",
        "original_upstream_channel_code": upstream / "RayTracing_cqi.py",
    }
    evidence = {name: source(path) for name, path in diagnostic_paths.items()}
    evidence["north_csv_byte_identical_to_upstream"] = evidence["static_north_csv"]["sha256"] == evidence["original_upstream_north_csv"]["sha256"]
    evidence["north_map_byte_identical_to_upstream"] = evidence["current_north_map"]["sha256"] == evidence["original_upstream_north_map"]["sha256"]
    evidence["channel_code_byte_identical_to_upstream"] = evidence["current_channel_code"]["sha256"] == evidence["original_upstream_channel_code"]["sha256"]
    metadata = dict(original_generation)
    metadata.update({
        "scenario_variant": "static_terminal_anchored_channel_mobility",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "time_origin": "t=0 exactly inherits original static positions, requests, labels and channel values.",
        "channel_model": "Per-user legacy-static channel anchor plus the existing deterministic geometric model increments.",
        "channel_anchor": {
            "snr_formula": "SNR_anchor(u,t) = SNR_static(u) + SNR_raw(u,t) - SNR_raw(u,0)",
            "rx_formula": "RX_anchor(u,t) = RX_static(u) + RX_raw(u,t) - RX_raw(u,0)",
            "cqi_formula": "round(1 + ((clamp(SNR_anchor,-10,30) + 10) / 40) * 14), Python round, matching upstream function",
            "scope": "All 90 users, all 31 frames. Fixed per-user offset; no artificial first-step channel switch.",
            "signal_precision": "Decimal arithmetic on the existing 6-decimal raw timeline and original 2-decimal static values; exact original channel text at t0 and for stationary users.",
            "los_semantics": "LOS at t0 and for stationary users is the original static LOS; at later nonstationary-profile frames it is current geometry LOS. raw_model_los always records current geometry, and exported_los_origin identifies provenance. Anchored SNR is not claimed to be a fresh uncalibrated path-loss prediction from exported LOS.",
            "dynamic_state_allocation": "No allocation/admission/classification chosen or changed by this dataset generator.",
        },
        "limitations": [
            "The legacy north CSV is internally SNR/CQI-consistent but cannot be reproduced at its source coordinates by the supplied current map/code settings; 17/30 CQI values differ before anchoring. Historical cause is not established.",
            "The per-user static offset is a continuity calibration. It preserves model-predicted changes along the existing path, not independently measured future channel values.",
            "The original static SNR/RX noise-floor rounding is preserved. SNR and RX are independently anchored to avoid changing any static channel field.",
            "Four original building-contained receivers remain stationary; inherited geometry is not a calibrated indoor propagation model.",
            "Source movement validation applies because positions, IDs, profiles, schedules and distance/speed fields are unchanged.",
            "This dataset does not itself execute a dynamic control experiment.",
        ],
        "exporter": source(Path(__file__)),
        "source_input_directory": str(args.input_dir.resolve()),
        "source_input_files": {str(path.relative_to(args.input_dir)): source(path)
                               for path in sorted(args.input_dir.rglob("*")) if path.is_file()},
        "static_sources": {scene: source(args.static_dir / f"{scene}.csv") for scene in SCENES},
        "north_source_diagnostic_evidence": evidence,
        "code_sha256": {"continuation/generate_anchored_channel.py": sha(Path(__file__))},
        "source_code_sha256": original_generation.get("code_sha256", {}),
        "scenarios": {scene: {**original_generation["scenarios"][scene], "statistics": reports[scene],
                               "channel_policy": "legacy-static per-user anchored increments"} for scene in SCENES},
        "output_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in outputs.items()},
    })
    outputs["generation.json"] = json_bytes(metadata)
    note = (
        "静态终态延续动态场景：信道衔接说明\n\n"
        "本数据在每个场景保留30名原始用户、原轨迹和静止/持续移动/间歇移动各10人的分组；共31帧、300秒、2790条记录。\n"
        "t=0逐用户继承静态CSV的CQI、SNR、接收功率、LOS、位置、需求与真实标签。静止用户整个300秒保持原静态信道；间歇用户暂停期间信道不变。\n\n"
        "衔接公式：SNR(u,t)=静态SNR(u)+原动态SNR(u,t)-原动态SNR(u,0)；接收功率按同样增量锚定。CQI继续使用原有SNR裁剪至[-10,30]再映射至1–15的公式。每用户偏移在全程固定，不在第10秒切换信道模型。\n"
        "原北部CSV与原始上游副本逐字节一致，地图和模型代码的副本也一致；原静态SNR与CQI相互自洽。但使用当前附带地图/模型对同一位置重算，北部17人的CQI发生差异，SNR最大差45.314624dB。没有证据确定历史生成参数，故不把差异归为位置移动。本版以静态信道为基准，仅应用已有轨迹带来的模型相对变化。\n\n"
        "CSV中的raw_model_snr_db/raw_model_rx_power_dbm/raw_model_cqi/raw_model_los保留未锚定模型值，channel_snr_offset_db/channel_rx_offset_db给出固定偏移。LOS在第0帧及静止用户全程取原静态值，其余帧取当前几何模型值；exported_los_origin明确来源。锚定后的SNR不是仅由该LOS重新计算的未经校准绝对预测。当前数据90人的原静态LOS与模型第0帧LOS本来就全部一致。\n\n"
        "该文件只准备输入，不重新接入用户、不纠正静态错误分类、不修改静态分配。原移动数据、原始实验结果与地图均未覆盖。\n"
        f"完整实验输入：{args.output.resolve()}\n"
        f"源移动输入：{args.input_dir.resolve()}\n"
        "复现：使用现有Python解释器运行 continuation/generate_anchored_channel.py，并指定尚不存在的--output与可写--package-dir；程序拒绝覆盖既有文件。\n"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    for name, content in outputs.items():
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(content)
    args.package_dir.mkdir(parents=True, exist_ok=True)
    with package_csv.open("xb") as handle:
        handle.write(outputs["dynamic_users_from_static.csv"])
    with package_note.open("x", encoding="utf-8") as handle:
        handle.write(note)
    for name, expected in metadata["output_sha256"].items():
        if sha(args.output / name) != expected:
            raise RuntimeError(f"Output checksum failed: {name}")
    for record in metadata["source_input_files"].values():
        if sha(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("Original input changed during export")
    print(json.dumps({"output": str(args.output.resolve()), "package_csv": str(package_csv.resolve()),
                      "validation": validation, "source_and_output_hashes_verified": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
