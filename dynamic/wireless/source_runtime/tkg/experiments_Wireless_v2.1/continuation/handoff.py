"""Validate immutable static-terminal evidence without optimizing or classifying."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_handoff(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, recorded in manifest["output_files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha256(path) != recorded["sha256"]:
            raise ValueError(f"Static handoff artifact changed or escaped its directory: {name}")
    scenes = {}
    for metadata in manifest["scenes"]:
        scene = metadata["scene"]
        if scene in scenes:
            raise ValueError("Duplicate static handoff scenario")
        with (directory / scene / "static_terminal_users.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        step = json.loads((directory / scene / "terminal_step.json").read_text(encoding="utf-8"))
        raw = json.loads((directory / scene / "raw_static_state.json").read_text(encoding="utf-8"))
        if raw != step["state"]:
            raise ValueError(f"{scene}: raw static state differs from terminal step")
        index = {str(row["user_id"]): row for row in rows}
        if len(index) != 30 or set(index) != {str(i) for i in range(1, 31)}:
            raise ValueError(f"{scene}: require the complete static population of 30")
        allocations = {str(row["user_id"]): row for row in step["allocations"]}
        if len(allocations) != len(step["allocations"]):
            raise ValueError(f"{scene}: duplicate original allocation")
        if set(allocations) != set(metadata["admitted_user_ids"]):
            raise ValueError(f"{scene}: source admission metadata differs")
        for uid, row in index.items():
            admitted = row["legacy_admitted"] == "1"
            if admitted != (uid in allocations):
                raise ValueError(f"{scene}/{uid}: source admission differs")
            if admitted:
                original = allocations[uid]
                for field, source in (("bandwidth_mhz", "bandwidth"), ("original_cqi", "cqi"),
                                      ("source_rate_mbps", "rate"), ("source_latency_ms", "latency")):
                    if float(row[field]) != float(original[source]):
                        raise ValueError(f"{scene}/{uid}: exported allocation {field} differs")
                if row["assigned_slice"] != original["slice"]:
                    raise ValueError(f"{scene}/{uid}: source assigned slice differs")
        scenes[scene] = {
            "rows": index, "terminal_step": step, "raw_static_state": raw,
            "provenance": {
                "source": "exact_static_terminal_import",
                "handoff_dir": str(directory),
                "handoff_manifest_sha256": sha256(manifest_path),
                "scene_metadata": deepcopy(metadata),
                "terminal_step_sha256": sha256(directory / scene / "terminal_step.json"),
                "raw_static_state_sha256": sha256(directory / scene / "raw_static_state.json"),
                "optimization_at_t0": False, "agent_executed_at_t0": False,
                "legacy_latency_semantics": "Preserved original static assigned value; not measured dynamic latency.",
                "static_traffic_history_imported": False,
            },
        }
    if set(scenes) != {"center", "north", "south"}:
        raise ValueError("Require all three original static scenarios")
    return scenes, {"directory": str(directory), "manifest_sha256": sha256(manifest_path),
                    "output_sha256": {k: v["sha256"] for k, v in manifest["output_files"].items()}}
