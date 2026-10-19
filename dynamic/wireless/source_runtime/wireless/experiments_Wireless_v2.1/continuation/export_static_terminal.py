#!/usr/bin/env python3
"""Export a read-only, policy-independent handoff of the static final states.

The exported input comparison does not choose a dynamic channel policy, alter
mobility, optimize allocations, correct classifications, or invoke any API.
Existing output directories are never overwritten. Python standard library only.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
from zoneinfo import ZoneInfo


SCENES = ("center", "north", "south")
ALLOCATION_FIELDS = ("user_id", "slice", "bandwidth", "cqi", "rate", "latency")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}


def terminal_step(path: Path) -> tuple[dict, int]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No recorded steps: {path}")
    value = json.loads(lines[-1])
    if len(lines) != 30 or value["arrivals"] != 30:
        raise ValueError(f"Expected the complete 30-arrival static trial: {path}")
    return value, len(lines)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def index(rows: list[dict], key: str) -> dict:
    result = {str(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate user identifiers in {key}")
    return result


def allocation_signature(step: dict) -> list[dict]:
    return [
        {key: allocation[key] for key in ALLOCATION_FIELDS}
        for allocation in sorted(step["allocations"], key=lambda item: int(item["user_id"]))
    ]


def physical_state_signature(step: dict) -> dict:
    state = json.loads(json.dumps(step["state"]))
    # The upstream KB serializer alone adds a wall-clock timestamp. It is not
    # a radio state variable and cannot equal another trial's execution time.
    state.pop("timestamp", None)
    for name in ("embb_slice", "urllc_slice"):
        state[name]["users"].sort(key=lambda item: int(item["user_id"]))
    return state


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def prepare_scene(scene: str, static_batch: Path, static_csv_dir: Path,
                  mobility_dir: Path) -> tuple[dict, dict[str, str]]:
    trial = static_batch / f"{scene}__tkg_dtn__r001"
    step_path = trial / "steps.jsonl"
    selected_path = trial / "selected_users.json"
    source_path = static_csv_dir / f"{scene}.csv"
    mobility_path = mobility_dir / f"{scene}.csv"
    step, line_number = terminal_step(step_path)
    selected_raw = json.loads(selected_path.read_text(encoding="utf-8"))
    selected = index(selected_raw, "user_id")
    source = index(read_csv(source_path), "RX_ID")
    frame0_rows = [row for row in read_csv(mobility_path)
                   if int(row["Frame_Index"]) == 0 and float(row["Time_s"]) == 0]
    frame0 = index(frame0_rows, "RX_ID")
    expected = {str(number) for number in range(1, 31)}
    if set(selected) != expected or set(frame0) != expected:
        raise ValueError(f"{scene}: expected all 30 static and frame-zero users")
    if not expected.issubset(source):
        raise ValueError(f"{scene}: missing original source users")
    allocations = index(step["allocations"], "user_id")
    if not set(allocations).issubset(expected):
        raise ValueError(f"{scene}: unknown admitted users")

    allocation_hash = canonical_digest(allocation_signature(step))
    physical_hash = canonical_digest(physical_state_signature(step))
    comparisons = []
    for method in ("tkg_dtn", "wireless_kb"):
        for repeat in range(1, 6):
            other_trial = static_batch / f"{scene}__{method}__r{repeat:03d}"
            other_path = other_trial / "steps.jsonl"
            other_step, other_line = terminal_step(other_path)
            other_selected_path = other_trial / "selected_users.json"
            other_selected = json.loads(other_selected_path.read_text(encoding="utf-8"))
            equal_allocations = allocation_signature(step) == allocation_signature(other_step)
            equal_state = physical_state_signature(step) == physical_state_signature(other_step)
            equal_population = selected_raw == other_selected
            if not (equal_allocations and equal_state and equal_population):
                raise ValueError(f"Static terminal mismatch: {other_trial}")
            comparisons.append({
                "method": method, "repeat": repeat,
                "steps_source": source_record(other_path),
                "terminal_line_1based": other_line,
                "selected_users_source": source_record(other_selected_path),
                "same_selected_users": equal_population,
                "same_allocation_fields": equal_allocations,
                "same_complete_physical_state": equal_state,
                "canonical_allocation_sha256": canonical_digest(allocation_signature(other_step)),
                "canonical_physical_state_sha256": canonical_digest(physical_state_signature(other_step)),
            })

    users, input_comparison = [], []
    for uid in sorted(expected, key=int):
        observed, original, mobile = selected[uid], source[uid], frame0[uid]
        for source_key, selected_key in (("X", "x"), ("Y", "y"), ("Z", "z"), ("CQI", "cqi")):
            if float(original[source_key]) != float(observed[selected_key]):
                raise ValueError(f"{scene}/{uid}: source {source_key} differs from static selected user")
        if original["User_Request"] != observed["request"] or original["Request_Label"] != observed["ground_truth"]:
            raise ValueError(f"{scene}/{uid}: original source request/label mismatch")
        allocation = allocations.get(uid)
        admitted = allocation is not None
        correct_slice = allocation["slice"] == observed["ground_truth"] if admitted else None
        if admitted and (allocation["cqi"] != observed["cqi"] or allocation["correct_slice"] != correct_slice):
            raise ValueError(f"{scene}/{uid}: allocation audit mismatch")
        users.append({
            "scene": scene, "user_id": uid, "legacy_admitted": int(admitted),
            "assigned_slice": allocation["slice"] if admitted else "",
            "bandwidth_mhz": allocation["bandwidth"] if admitted else 0,
            "original_cqi": observed["cqi"], "original_snr_db": original["SNR_dB"],
            "original_rx_power_dbm": original["RX_Power_dBm"], "original_los": original["LOS"],
            "original_x": original["X"], "original_y": original["Y"], "original_z": original["Z"],
            "ground_truth": observed["ground_truth"], "request": observed["request"],
            "correct_slice": int(correct_slice) if admitted else "",
            "source_rate_mbps": allocation["rate"] if admitted else 0,
            "source_latency_ms": allocation["latency"] if admitted else "",
            "legacy_resource_feasible": int(allocation["resource_feasible"]) if admitted else "",
            "legacy_status": "admitted" if admitted else "rejected",
        })
        comparison = {"scene": scene, "user_id": uid}
        for field in ("X", "Y", "Z", "CQI", "SNR_dB", "RX_Power_dBm", "LOS"):
            comparison[f"static_{field}"] = original[field]
            comparison[f"dynamic_frame0_{field}"] = mobile[field]
            comparison[f"delta_{field}"] = float(mobile[field]) - float(original[field])
        comparison["same_position"] = int(all(float(original[key]) == float(mobile[key]) for key in ("X", "Y", "Z")))
        comparison["same_request"] = int(original["User_Request"] == mobile["User_Request"])
        comparison["same_ground_truth"] = int(original["Request_Label"] == mobile["Request_Label"])
        comparison["same_cqi"] = int(int(original["CQI"]) == int(mobile["CQI"]))
        comparison["channel_policy_selected"] = "no"
        if not (comparison["same_position"] and comparison["same_request"] and comparison["same_ground_truth"]):
            raise ValueError(f"{scene}/{uid}: static-linked trajectory identity mismatch")
        input_comparison.append(comparison)

    summary = json.loads((trial / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or len(allocations) != step["admitted_users"]:
        raise ValueError(f"{scene}: incomplete/inconsistent original trial")
    files = {
        f"{scene}/raw_static_state.json": json_text(step["state"]),
        f"{scene}/terminal_step.json": json_text(step),
        f"{scene}/selected_users.json": json_text(selected_raw),
        f"{scene}/static_terminal_users.csv": csv_text(users),
        f"{scene}/dynamic_frame0_input_comparison.csv": csv_text(input_comparison),
    }
    metadata = {
        "scene": scene, "source_method": "tkg_dtn", "source_repeat": 1,
        "terminal_line_1based": line_number,
        "sources": {
            "steps": source_record(step_path), "selected_users": source_record(selected_path),
            "static_csv": source_record(source_path), "summary": source_record(trial / "summary.json"),
            "graph_snapshot": source_record(trial / "graph/graph_snapshot.json"),
            "dynamic_mobility_csv": source_record(mobility_path),
        },
        "observed_user_ids": sorted(expected, key=int),
        "admitted_user_ids": sorted(allocations, key=int),
        "rejected_user_ids": sorted(expected - set(allocations), key=int),
        "wrong_slice_user_ids": [row["user_id"] for row in users if row["correct_slice"] == 0],
        "original_final_metrics": summary["final"],
        "canonical_allocation_sha256": allocation_hash,
        "canonical_physical_state_sha256": physical_hash,
        "terminal_equivalence": {
            "reference": f"{scene}__tkg_dtn__r001", "checked_trials": len(comparisons),
            "allocation_fields": list(ALLOCATION_FIELDS),
            "physical_state_scope": "State with slice-user lists sorted by numeric user ID; only the upstream wall-clock timestamp is omitted. Raw tkg_dtn source state is exported unchanged.",
            "all_same": True, "trials": comparisons,
        },
        "frame0_comparison": {
            "compared_users": 30, "all_same_position_request_label": True,
            "cqi_mismatch_ids": [row["user_id"] for row in input_comparison if not row["same_cqi"]],
            "snr_mismatch_over_0_01_db_ids": [row["user_id"] for row in input_comparison if abs(row["delta_SNR_dB"]) > 0.01],
            "channel_policy_selected": False,
            "interpretation": "Input reconciliation only. Static state and dynamic frame-zero channel values are recorded separately; neither is replaced or adopted for a dynamic trial.",
        },
    }
    return metadata, files


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-batch", type=Path, default=project.parent / "experiments_Wireless_v2/results/four_methods_5repeats_20261012_212809")
    parser.add_argument("--static-csv-dir", type=Path, default=project.parent / "experiments_Wireless_v2/data/scenarios")
    parser.add_argument("--mobility-dir", type=Path, default=project / "data/mobility_5min_from_static_20261014")
    parser.add_argument("--output", type=Path, default=project / "data/static_terminal_handoff_20261014")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing handoff: {args.output}")
    metadata, outputs = [], {}
    for scene in SCENES:
        scene_metadata, scene_outputs = prepare_scene(scene, args.static_batch, args.static_csv_dir, args.mobility_dir)
        metadata.append(scene_metadata)
        outputs.update(scene_outputs)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "purpose": "Policy-independent exact static-terminal handoff evidence for all 30 observed users per scene.",
        "exporter": source_record(Path(__file__)),
        "immutable_source_files": True,
        "api_calls": 0,
        "dynamic_experiment_executed": False,
        "dynamic_channel_policy_selected": False,
        "semantics": {
            "population": "All 30 observed users remain in the export, including legacy rejected users.",
            "allocations": "Raw static final allocation preserved, including incorrect slices; no allocation optimization or reclassification.",
            "labels": "ground_truth is the original Request_Label, not the assigned slice; no contract/channel policy is selected by this exporter.",
            "rejected_users": "No allocation: assigned_slice/correct_slice/source_latency_ms are blank; bandwidth and realized source rate are zero.",
            "frame0": "Comparison is input checking only. It does not modify the source trajectory or specify how to reconcile differing channels.",
            "terminal_equivalence": "Compare all 5 TKG-DTN trials and all 5 wireless_kb trials per scene using exact numeric values, complete physical state, and full selected population.",
        },
        "scenes": metadata,
        "output_files": {name: {"sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(), "bytes": len(value.encode("utf-8"))}
                         for name, value in outputs.items()},
    }
    # Complete all source checks before creating any output. mkdir is exclusive.
    args.output.mkdir(parents=True, exist_ok=False)
    for relative, content in outputs.items():
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
    with (args.output / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(json_text(manifest))
    for relative, recorded in manifest["output_files"].items():
        if digest(args.output / relative) != recorded["sha256"]:
            raise RuntimeError(f"Output checksum mismatch: {relative}")
    print(json_text({"output": str(args.output.resolve()), "scenes": [
        {"scene": item["scene"], "observed": len(item["observed_user_ids"]),
         "admitted": len(item["admitted_user_ids"]), "rejected": item["rejected_user_ids"],
         "wrong_slice": item["wrong_slice_user_ids"],
         "identical_terminal_trials": item["terminal_equivalence"]["checked_trials"],
         "frame0_cqi_mismatches": len(item["frame0_comparison"]["cqi_mismatch_ids"])}
        for item in metadata], "output_checksum_validation": "passed",
        "api_calls": 0, "dynamic_channel_policy_selected": False}))


if __name__ == "__main__":
    main()
