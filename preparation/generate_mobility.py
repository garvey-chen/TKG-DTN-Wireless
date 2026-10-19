"""Generate time-indexed user mobility and deterministic upstream channel values."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
import platform
import tempfile
from collections import Counter
from pathlib import Path

import generate_scenarios as source
from mobility import simulate_users

ROOT = Path(__file__).resolve().parent
LONG_FIELDS = ["Scenario", "Frame_Index", "Time_s", *source.FIELDS,
               "Mobility_Profile", "Movement_State", "Step_Distance_m", "Speed_mps",
               "Cumulative_Distance_m"]
PROFILE_FIELDS = ["Scenario", "RX_ID", "Mobility_Profile", "Nominal_Speed_mps"]
USER_SUMMARY_FIELDS = ["Scenario", "RX_ID", "Mobility_Profile", "Total_Distance_m",
                       "Moving_Intervals", "Paused_Intervals", "Blocked_Intervals",
                       "CQI_Initial", "CQI_Final", "CQI_Min", "CQI_Max", "CQI_Changes"]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_initial(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != source.FIELDS:
            raise ValueError(f"Unexpected initial CSV columns: {path}")
        rows = list(reader)
    seen = set()
    for row in rows:
        uid = row["RX_ID"]
        if not uid or uid in seen or row["Request_Label"] not in ("eMBB", "URLLC"):
            raise ValueError(f"Invalid initial ID or request label in {path}: {uid}")
        if not row["User_Request"].strip():
            raise ValueError(f"Empty request in {path}: {uid}")
        if not all(math.isfinite(float(row[key])) for key in ("X", "Y", "Z", "SNR_dB", "RX_Power_dBm")):
            raise ValueError(f"Non-finite initial data in {path}: {uid}")
        if not 1 <= int(row["CQI"]) <= 15 or row["LOS"] not in ("0", "1"):
            raise ValueError(f"Invalid initial CQI/LOS in {path}: {uid}")
        seen.add(uid)
    if not rows:
        raise ValueError(f"Empty initial CSV: {path}")
    return rows


def calculate_channel(model, tx, position, buildings):
    los = model.has_line_of_sight(tx, position, buildings)
    loss = model.calculate_path_loss(tx, position, source.FREQUENCY_HZ, los)
    power = source.TX_POWER_DBM - loss
    snr = power - source.NOISE_FLOOR_DBM
    return {"SNR_dB": f"{snr:.6f}", "RX_Power_dBm": f"{power:.6f}",
            "CQI": str(model.calculate_cqi(snr)), "LOS": str(int(los))}


def generate_scene(model, scene, args, output):
    initial_path = args.initial_dir / f"{scene}.csv"
    initial = load_initial(initial_path)
    initial_by_id = {row["RX_ID"]: row for row in initial}
    buildings, tx, origin, bounds = source.scene_geometry(model, scene)
    scene_seed = args.seed + source.SCENE_OFFSETS[scene]
    frames, profiles = simulate_users(
        initial, buildings, bounds, duration_s=args.duration_s, interval_s=args.interval_s,
        seed=scene_seed, moving_fraction=args.moving_fraction,
        intermittent_fraction=args.intermittent_fraction,
        speed_min_mps=args.speed_min_mps, speed_max_mps=args.speed_max_mps,
    )
    all_rows = []
    for frame_index, frame in enumerate(frames):
        snapshot = []
        for movement in frame:
            original = initial_by_id[str(movement["RX_ID"])]
            position = tuple(float(movement[k]) for k in ("X", "Y", "Z"))
            row = {"RX_ID": original["RX_ID"],
                   **{key: f"{value:.6f}" for key, value in zip(("X", "Y", "Z"), position)},
                   **calculate_channel(model, tx, position, buildings),
                   "User_Request": original["User_Request"], "Request_Label": original["Request_Label"]}
            if frame_index == 0 and row != original:
                raise ValueError(
                    f"{scene} user {row['RX_ID']}: initial CSV must match the current map/model "
                    "at six-decimal precision; regenerate the initial scenario before mobility."
                )
            snapshot.append(row)
            all_rows.append({
                "Scenario": scene, "Frame_Index": frame_index, "Time_s": frame_index * args.interval_s,
                **row, "Mobility_Profile": movement["Mobility_Profile"],
                "Movement_State": movement["Movement_State"],
                **{key: f"{movement[key]:.6f}" for key in
                   ("Step_Distance_m", "Speed_mps", "Cumulative_Distance_m")},
            })
        write_csv(output / "frames" / f"frame_{frame_index:03d}" / f"{scene}.csv", source.FIELDS, snapshot)
    write_csv(output / f"{scene}.csv", LONG_FIELDS, all_rows)
    user_summaries = []
    for original in initial:
        selected = [row for row in all_rows if row["RX_ID"] == original["RX_ID"]]
        cqis = [int(row["CQI"]) for row in selected]
        states = Counter(row["Movement_State"] for row in selected[1:])
        user_summaries.append({
            "Scenario": scene, "RX_ID": original["RX_ID"],
            "Mobility_Profile": selected[0]["Mobility_Profile"],
            "Total_Distance_m": selected[-1]["Cumulative_Distance_m"],
            "Moving_Intervals": states["moving"], "Paused_Intervals": states["paused"],
            "Blocked_Intervals": states["blocked"], "CQI_Initial": cqis[0], "CQI_Final": cqis[-1],
            "CQI_Min": min(cqis), "CQI_Max": max(cqis),
            "CQI_Changes": sum(a != b for a, b in zip(cqis, cqis[1:])),
        })
    statistics = {
        "users": len(initial), "frames": len(frames), "rows": len(all_rows),
        "profile_counts": dict(Counter(p["mobility_profile"] for p in profiles)),
        "users_with_cqi_changes": sum(row["CQI_Changes"] > 0 for row in user_summaries),
        "cqi_change_events": sum(row["CQI_Changes"] for row in user_summaries),
        "cqi_range": [min(int(r["CQI"]) for r in all_rows), max(int(r["CQI"]) for r in all_rows)],
        "distance_range_m": [min(float(r["Total_Distance_m"]) for r in user_summaries),
                             max(float(r["Total_Distance_m"]) for r in user_summaries)],
        "blocked_intervals": sum(row["Blocked_Intervals"] for row in user_summaries),
    }
    metadata = {
        "seed": scene_seed, "initial_csv": str(initial_path),
        "initial_csv_sha256": source.sha256(initial_path),
        "map_file": f"data/maps/HKUST_{scene.title()}.osm",
        "map_sha256": source.sha256(ROOT / "data/maps" / f"HKUST_{scene.title()}.osm"),
        "tx_position_m": tx, "geographic_origin_lat_lon": origin,
        "sampling_bounds_xy_m": bounds, "bounds_order": ["min_x", "max_x", "min_y", "max_y"],
        "statistics": statistics, "user_profiles": profiles,
    }
    return {"scene": scene, "buildings": buildings, "tx": tx, "bounds": bounds,
            "rows": all_rows, "metadata": metadata, "user_summaries": user_summaries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=source.SCENES, default=list(source.SCENES))
    parser.add_argument("--duration-s", type=int, default=300)
    parser.add_argument("--interval-s", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--moving-fraction", type=float, default=2 / 3,
                        help="share of users that may move (default: 2/3)")
    parser.add_argument("--intermittent-fraction", type=float, default=0.5,
                        help="share of moving users with scheduled pauses (default: 1/2)")
    parser.add_argument("--speed-min-mps", type=float, default=0.4)
    parser.add_argument("--speed-max-mps", type=float, default=0.8)
    parser.add_argument("--initial-dir", type=Path, default=ROOT / "data/scenarios")
    parser.add_argument("--output", type=Path, default=ROOT / "data/mobility_5min")
    parser.add_argument("--no-plots", action="store_true", help="generate data without the optional plots")
    args = parser.parse_args(argv)
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Duplicate scenarios are not allowed")
    if args.duration_s <= 0 or args.interval_s <= 0 or args.duration_s % args.interval_s:
        parser.error("duration and interval must be positive; duration must be divisible by interval")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (args.moving_fraction, args.intermittent_fraction)):
        parser.error("fractions must be finite and between 0 and 1")
    if not (math.isfinite(args.speed_min_mps) and math.isfinite(args.speed_max_mps)
            and 0 < args.speed_min_mps <= args.speed_max_mps):
        parser.error("speeds must be finite with 0 < min <= max")
    output = args.output.resolve()
    args.initial_dir = args.initial_dir.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    model = source.load_model()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mobility_generation_", dir=output.parent) as tmp:
        staging = Path(tmp) / "output"
        staging.mkdir()
        scenes = [generate_scene(model, scene, args, staging) for scene in args.scenarios]
        timeline = []
        for index, seconds in enumerate(range(0, args.duration_s + 1, args.interval_s)):
            directory = f"frames/frame_{index:03d}"
            timeline.append({"Frame_Index": index, "Time_s": seconds,
                             "Time_Label": f"{seconds // 60:02d}:{seconds % 60:02d}",
                             "Frame_Directory": directory,
                             **{f"{scene}_csv": f"{directory}/{scene}.csv" for scene in args.scenarios}})
        write_csv(staging / "timeline.csv", list(timeline[0]), timeline)
        profile_rows = [{"Scenario": scene["scene"], "RX_ID": p["user_id"],
                         "Mobility_Profile": p["mobility_profile"],
                         "Nominal_Speed_mps": f'{p["nominal_speed_mps"]:.6f}'}
                        for scene in scenes for p in scene["metadata"]["user_profiles"]]
        write_csv(staging / "user_profiles.csv", PROFILE_FIELDS, profile_rows)
        write_csv(staging / "user_summary.csv", USER_SUMMARY_FIELDS,
                  [row for scene in scenes for row in scene["user_summaries"]])
        if not args.no_plots:
            from plot_mobility import plot_trajectories
            plot_trajectories(scenes, staging)
        metadata = {
            "duration_s": args.duration_s, "interval_s": args.interval_s,
            "frame_count": len(timeline), "time_origin": "relative simulation time; initial state at t=0 s",
            "interval_semantics": "Frame at t records position/channel at t and movement during (t-dt, t]; initial distances/speeds are zero.",
            "base_seed": args.seed, "scene_seed_rule": "base_seed + {center:0,north:1,south:2}[scene]",
            "moving_fraction": args.moving_fraction, "intermittent_fraction_of_movers": args.intermittent_fraction,
            "nominal_speed_range_mps": [args.speed_min_mps, args.speed_max_mps],
            "movement_model": "Correlated random walk with full-segment building avoidance and scheduled pauses; no teleportation.",
            "channel_model": "Same upstream map/LoS/path-loss/SNR/CQI functions as the static v2.1 initial data; no random fading.",
            "channel_parameters": {"frequency_hz": source.FREQUENCY_HZ,
                                   "tx_power_dbm": source.TX_POWER_DBM,
                                   "noise_bandwidth_hz": source.BANDWIDTH_HZ,
                                   "noise_figure_db": source.NOISE_FIGURE_DB,
                                   "noise_floor_dbm": source.NOISE_FLOOR_DBM},
            "user_identity": "Use (Scenario,RX_ID) as the persistent user key; requests/labels do not change or arrive again each frame.",
            "coordinate_and_signal_decimals": 6,
            "limitations": [
                "No inter-user collision avoidance; the initial 5 m spacing is not a movement constraint.",
                "Building geometry is enforced, but roads, shorelines and land-use restrictions are not modeled.",
                "The upstream LoS check includes the TX-supporting building, as in the initial data.",
                "Each scene has its own coordinate origin; users do not move between scenes.",
                "This is movement/channel input only, not a network control or handover result.",
            ],
            "code_sha256": {name: source.sha256(ROOT / name) for name in
                            ("generate_mobility.py", "mobility.py", "plot_mobility.py",
                             "generate_scenarios.py", "vendor/upstream/RayTracing_cqi.py")},
            "software": {"python": platform.python_version(), **{
                name: importlib.metadata.version(name) for name in ("numpy", "matplotlib", "pyproj", "shapely")}},
            "scenarios": {scene["scene"]: scene["metadata"] for scene in scenes},
            "output_sha256": {str(p.relative_to(staging)): source.sha256(p)
                              for p in sorted(staging.rglob("*")) if p.is_file()},
        }
        source.write_json(staging / "generation.json", metadata)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
        staging.rename(output)
    for scene in scenes:
        stats = scene["metadata"]["statistics"]
        print(f"{scene['scene']}: {stats['users']} users x {stats['frames']} frames; "
              f"profiles={stats['profile_counts']}; CQI changed for {stats['users_with_cqi_changes']} users")
    print(f"Saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
