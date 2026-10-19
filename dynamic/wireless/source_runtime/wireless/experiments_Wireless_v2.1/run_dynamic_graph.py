"""Replay mobility frames into a persistent temporal graph, without agent actions."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile

from dynamic_tkg.graph import TemporalWirelessGraph, packed
from dynamic_tkg.export import export_graph
from dynamic_tkg.service import INITIALIZATION_MODES, INITIALIZATION_DESCRIPTIONS

ROOT = Path(__file__).resolve().parent
SCENES = ("center", "north", "south")
METRIC_FIELDS = ["scenario", "frame_index", "time_s", "revision", "observed_users", "admitted_users",
                 "effective_users", "degraded_users", "unadmitted_users", "raw_throughput_mbps",
                 "effective_throughput_mbps", "jain_rate_all_admitted", "jain_demand_normalized_all_admitted",
                 "mean_rate_satisfaction_clipped_all_admitted", "embb_used_mhz", "urllc_used_mhz", "latency_assessed"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: packed(row.get(k)) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in fields})


def summarize(states, events):
    result = {}
    for scene in sorted({s["scenario"] for s in states}):
        selected = sorted((s for s in states if s["scenario"] == scene), key=lambda s: s["time_s"])
        start, final = selected[0], selected[-1]
        duration = final["time_s"] - start["time_s"]
        metrics = [s["metrics"] for s in selected]
        user_seconds = sum((b["time_s"] - a["time_s"]) * a["metrics"]["effective_users"] for a, b in zip(selected, selected[1:]))
        denominator = duration * start["metrics"]["admitted_users"]
        result[scene] = {
            "initial_user_ids": list(start["allocations"]), "admitted_users": start["metrics"]["admitted_users"],
            "observed_users": start["metrics"]["observed_users"], "frame_count": len(selected), "duration_s": duration,
            "effective_users_min": min(m["effective_users"] for m in metrics),
            "effective_users_max": max(m["effective_users"] for m in metrics),
            "effective_users_final": final["metrics"]["effective_users"],
            "raw_throughput_range_mbps": [min(m["raw_throughput_mbps"] for m in metrics), max(m["raw_throughput_mbps"] for m in metrics)],
            "effective_throughput_range_mbps": [min(m["effective_throughput_mbps"] for m in metrics), max(m["effective_throughput_mbps"] for m in metrics)],
            "effective_user_seconds": user_seconds,
            "effective_user_time_ratio": user_seconds / denominator if denominator else None,
            "duration_convention": "Forward hold on [t_i,t_(i+1)); final sample has zero additional duration.",
            "event_counts": dict(Counter(e["event_type"] for e in events if e["scenario"] == scene)),
            "episodes": final["episodes"], "end_open_episodes": len(final["active_episodes"]),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/mobility_5min")
    parser.add_argument("--output", type=Path, default=ROOT / "results/dynamic_graph_5min")
    parser.add_argument("--initial-users", type=int, default=10)
    parser.add_argument("--initialization-mode", choices=INITIALIZATION_MODES,
                        default="feasible_admission",
                        help="feasible_admission requires all selected users effective at t=0; fixed_population retains the first N users and records unmet service minima")
    parser.add_argument("--scenarios", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--rate-drop-fraction", type=float, default=0.1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.initial_users < 1 or len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Use positive --initial-users and unique --scenarios")
    if not 0 < args.rate_drop_fraction <= 1:
        parser.error("--rate-drop-fraction must be in (0,1]")
    output, input_dir = args.output.resolve(), args.input_dir.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    timeline = read_csv(input_dir / "timeline.csv")
    if not timeline:
        parser.error("Empty timeline")
    for index, row in enumerate(timeline):
        if int(row["Frame_Index"]) != index or float(row["Time_s"]) < 0:
            parser.error("Timeline must start at frame 0 and be consecutive")
    generation_path = input_dir / "generation.json"
    input_info = json.loads(generation_path.read_text(encoding="utf-8")) if generation_path.exists() else {}
    input_hashes = {"timeline.csv": sha256(input_dir / "timeline.csv")}
    if generation_path.exists():
        input_hashes["generation.json"] = sha256(generation_path)
    source_paths = [Path(__file__), *sorted((ROOT / "dynamic_tkg").glob("*.py"))]
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dynamic_graph_", dir=output.parent) as tmp:
        staging = Path(tmp) / "output"
        staging.mkdir()
        for source in source_paths:
            target = staging / "source_snapshot" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        graph = TemporalWirelessGraph(staging / "graph.sqlite")
        states, metrics, events, episodes = [], [], [], []
        try:
            for index, timing in enumerate(timeline):
                time_s = float(timing["Time_s"])
                for scene in args.scenarios:
                    path = (input_dir / timing[f"{scene}_csv"]).resolve()
                    if not path.is_relative_to(input_dir):
                        raise ValueError("Snapshot paths must remain inside the selected input directory")
                    input_hashes[str(path.relative_to(input_dir))] = sha256(path)
                    meta = input_info.get("scenarios", {}).get(scene, {})
                    state = graph.sync_frame(scene, index, time_s, read_csv(path), expected_revision=index,
                                             target_users=args.initial_users, rate_drop_fraction=args.rate_drop_fraction,
                                             initialization_mode=args.initialization_mode,
                                             scene_info={key: meta[key] for key in
                                                         ("tx_position_m", "geographic_origin_lat_lon", "map_sha256") if key in meta})
                    states.append(state)
                    m = state["metrics"]
                    metrics.append({"scenario": scene, "frame_index": index, "time_s": time_s, "revision": state["revision"],
                                    **{k: m[k] for k in METRIC_FIELDS if k in m},
                                    "embb_used_mhz": m["slice_usage_mhz"]["eMBB"], "urllc_used_mhz": m["slice_usage_mhz"]["URLLC"]})
            for scene in args.scenarios:
                events.extend(graph.history(scene))
                episodes.extend(graph.current_state(scene)["episodes"])
                write_json(staging / f"latest_context_{scene}.json", graph.read_context(scene))
            run_id = graph.run_id
        finally:
            graph.close()
        with (staging / "frames.jsonl").open("w", encoding="utf-8") as handle:
            for state in states:
                handle.write(packed(state) + "\n")
        for name, rows in (("service_events", events), ("episodes", episodes)):
            with (staging / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(packed(row) + "\n")
        write_csv(staging / "frame_metrics.csv", METRIC_FIELDS, metrics)
        write_csv(staging / "service_events.csv", ["event_id", "scenario", "frame_index", "time_s", "revision",
                  "user_id", "event_type", "episode_id", "before", "after", "reason"], events)
        write_csv(staging / "episodes.csv", ["episode_id", "scenario", "user_id", "start_time_s", "end_time_s",
                  "last_observed_time_s", "observed_duration_s", "is_open", "reason", "onset_event_id", "recovery_event_id"], episodes)
        export = export_graph(staging / "graph.sqlite", staging)
        # Store portable artifact names, not the temporary staging path.
        export["artifacts"] = {name: name for name in export["artifacts"]}
        if not args.no_plots:
            from dynamic_tkg.plot import plot_monitoring
            plot_monitoring(metrics, staging)
        summary = summarize(states, events)
        write_json(staging / "summary.json", summary)
        for name, digest in code_hashes.items():
            if sha256(ROOT / name) != digest or sha256(staging / "source_snapshot" / name) != digest:
                raise AssertionError(f"Source changed during baseline run: {name}")
        manifest = {
            "run_id": run_id, "backend": "sqlite_local_property_graph", "neo4j_import_executed": False,
            "agent_executed": False, "mode": "fixed_initial_allocation_monitoring",
            "initial_users_per_scene": args.initial_users, "rate_drop_fraction": args.rate_drop_fraction,
            "initialization_mode": args.initialization_mode,
            "initialization": INITIALIZATION_DESCRIPTIONS[args.initialization_mode],
            "initialization_summary": {s["scenario"]: {
                "selected_user_ids": list(s["allocations"]),
                "effective_users": s["metrics"]["effective_users"],
                "degraded_user_ids": list(s["active_episodes"]),
                "evidence": s.get("initialization_evidence"),
            } for s in states if s["frame_index"] == 0},
            "objective": ("Freeze the initial count-first fair numerical allocation throughout replay; no subsequent optimization."
                          if args.initialization_mode == "fixed_population" else
                          "Primary: maximize modeled effective users in the fixed cohort; secondary: effective throughput. Fairness is reported, not optimized in this monitoring replay."),
            "limitations": "No delay measurements, control decisions, admission changes or automatic fault repair. Numeric preview retains the full cohort and may correctly report infeasible.",
            "input_dir": str(input_dir), "input_sha256": input_hashes,
            "code_sha256": code_hashes,
            "source_snapshot_layout": "source_snapshot/<code_sha256 relative path>",
            "python": platform.python_version(), "graph": export,
            "output_sha256": {str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()},
        }
        write_json(staging / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    print(f"Saved local temporal graph: {output}")
    print(f"{export['node_count']} nodes, {export['edge_count']} edges; {len(states)} frames, {len(events)} change events")
    for scene, values in summary.items():
        print(f"{scene}: admitted={values['admitted_users']}, effective={values['effective_users_min']}..{values['effective_users_max']}, "
              f"open episodes={values['end_open_episodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
