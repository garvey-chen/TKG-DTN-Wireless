"""Generate location variants with the supplied WirelessAgent propagation code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import random
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCENES = ("center", "north", "south")
SCENE_OFFSETS = {"center": 0, "north": 1, "south": 2}
FIELDS = ["RX_ID", "X", "Y", "Z", "SNR_dB", "RX_Power_dBm",
          "CQI", "LOS", "User_Request", "Request_Label"]
FREQUENCY_HZ = 2.4e9
BANDWIDTH_HZ = 20e6
NOISE_FIGURE_DB = 8
TX_POWER_DBM = 30
MIN_DISTANCE_M = 5.0
NOISE_FLOOR_DBM = -174 + 10 * math.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_model():
    # Import only the supplied scene model; no LLM, database or sibling imports.
    import matplotlib
    matplotlib.use("Agg")
    path = ROOT / "vendor/upstream/RayTracing_cqi.py"
    spec = importlib.util.spec_from_file_location("wireless_v21_upstream", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_geometry(model, scene: str):
    path = ROOT / "data/maps" / f"HKUST_{scene.title()}.osm"
    buildings = model.parse_osm_buildings(str(path))
    buildings, _, origin = model.convert_to_cartesian(buildings)
    tallest = model.find_tallest_building(buildings)
    if tallest is None:
        raise ValueError(f"No buildings in {path}")
    tx = model.place_tx(tallest)
    nodes = [node for building in buildings for node in building["nodes"]]
    xs, ys = zip(*nodes)
    buffer = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.1
    bounds = (min(xs) - buffer, max(xs) + buffer, min(ys) - buffer, max(ys) + buffer)
    return buildings, tx, origin, bounds


def read_template(scene: str) -> list[dict]:
    path = ROOT / "data/source_scenarios" / f"{scene}.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected columns in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty source scenario: {path}")
    for index, row in enumerate(rows, 1):
        if (row["RX_ID"] != str(index) or not row["User_Request"].strip()
                or row["Request_Label"] not in ("eMBB", "URLLC")):
            raise ValueError(f"Invalid source request/label/ID in {path}, row {index}")
    return rows


def check_positions(model, positions, buildings, bounds, templates) -> None:
    """Validate the coordinates actually serialized to CSV, before propagation."""
    min_x, max_x, min_y, max_y = bounds
    for index, (x, y, z) in enumerate(positions):
        if not all(math.isfinite(v) for v in (x, y, z)) or z != 1.5:
            raise ValueError("Invalid receiver position")
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            raise ValueError("Rounded receiver lies outside sampling bounds; use another seed")
        if any(b["polygon"].contains(model.Point(x, y)) for b in buildings):
            raise ValueError("Rounded receiver lies inside a building; use another seed")
        if any(math.hypot(x - p[0], y - p[1]) < MIN_DISTANCE_M for p in positions[:index]):
            raise ValueError("Rounded receivers are less than 5 m apart; use another seed")
        old = templates[index % len(templates)]
        if (x, y) == (float(old["X"]), float(old["Y"])):
            raise ValueError("A new receiver equals its source position; use another seed")


def generate_scene(model, scene: str, users: int | str, seed: int, output: Path):
    templates = read_template(scene)
    count = len(templates) if users == "source" else users
    buildings, tx, origin, bounds = scene_geometry(model, scene)
    scene_seed = seed + SCENE_OFFSETS[scene]
    # A separate stream per scene makes selection/order independent and avoids
    # changing Python's global RNG. No random fading is present in this model.
    model.random = random.Random(scene_seed)
    raw_positions = model.generate_rx_positions(bounds, count, buildings, MIN_DISTANCE_M)
    if len(raw_positions) != count:
        raise ValueError(f"{scene}: placed only {len(raw_positions)} of {count} users")
    positions = [tuple(round(v, 6) for v in p) for p in raw_positions]
    check_positions(model, positions, buildings, bounds, templates)
    results = model.perform_ray_tracing(tx, positions, buildings, tx_power_dBm=TX_POWER_DBM)

    rows, trace = [], []
    for index, item in enumerate(results):
        source_index = index % len(templates)
        source = templates[source_index]
        row = {
            "RX_ID": str(index + 1),
            **{key: f"{value:.6f}" for key, value in zip(("X", "Y", "Z"), item["position"])},
            "SNR_dB": f'{item["snr_dB"]:.6f}',
            "RX_Power_dBm": f'{item["rx_power_dBm"]:.6f}',
            "CQI": item["cqi"], "LOS": int(item["has_los"]),
            # Discard the upstream random request; keep the source pair intact.
            "User_Request": source["User_Request"], "Request_Label": source["Request_Label"],
        }
        rows.append(row)
        trace.append({
            "rx_id": index + 1, "source_row_1based": source_index + 1,
            "source_rx_id": source["RX_ID"],
            "source_position_m": [float(source[k]) for k in ("X", "Y", "Z")],
            "source_cqi": int(source["CQI"]),
            "distance_to_tx_m": math.dist(tx, item["position"]),
            "path_loss_db": TX_POWER_DBM - item["rx_power_dBm"],
            "snr_db_unrounded": item["snr_dB"],
        })

    csv_path = output / f"{scene}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    cqi_values = [row["CQI"] for row in rows]
    stats = {
        "users": count,
        "label_counts": dict(sorted(Counter(row["Request_Label"] for row in rows).items())),
        "cqi_min": min(cqi_values), "cqi_max": max(cqi_values),
        "cqi_mean": sum(cqi_values) / count,
        "cqi_histogram": {str(k): cqi_values.count(k) for k in range(1, 16)},
        "los_users": sum(row["LOS"] for row in rows),
        "changed_positions": count,
        "changed_cqi_vs_source": sum(row["CQI"] != t["source_cqi"] for row, t in zip(rows, trace)),
    }
    metadata = {
        "scenario": scene, "base_seed": seed, "scene_seed": scene_seed,
        "seed_rule": "base_seed + {center: 0, north: 1, south: 2}[scene]",
        "request_policy": "source order; exact text/label pairs; cyclic reuse only beyond source row count",
        "source_users": len(templates),
        "request_cycles_used": math.ceil(count / len(templates)),
        "map_file": f"data/maps/HKUST_{scene.title()}.osm",
        "map_sha256": sha256(ROOT / "data/maps" / f"HKUST_{scene.title()}.osm"),
        "source_csv_file": f"data/source_scenarios/{scene}.csv",
        "source_csv_sha256": sha256(ROOT / "data/source_scenarios" / f"{scene}.csv"),
        "csv_sha256": sha256(csv_path),
        "generator_sha256": sha256(Path(__file__)),
        "upstream_model_sha256": sha256(ROOT / "vendor/upstream/RayTracing_cqi.py"),
        "model": "Unmodified upstream simplified LoS/NLoS path loss; no multipath or random fading",
        "tx_position_m": tx, "geographic_origin_lat_lon": origin,
        "sampling_bounds_xy_m": bounds, "bounds_order": ["min_x", "max_x", "min_y", "max_y"],
        "buildings": len(buildings), "boundary_buffer_fraction": 0.1,
        "receiver_height_m": 1.5, "min_receiver_spacing_m": MIN_DISTANCE_M,
        "coordinate_decimals": 6, "signal_decimals": 6,
        "precision_policy": "Round coordinates before propagation; calculate CQI from unrounded SNR",
        "frequency_hz": FREQUENCY_HZ, "noise_bandwidth_hz": BANDWIDTH_HZ,
        "noise_figure_db": NOISE_FIGURE_DB, "tx_power_dbm": TX_POWER_DBM,
        "noise_floor_dbm": NOISE_FLOOR_DBM,
        "known_upstream_limits": [
            "The TX-supporting building participates in the upstream LoS obstruction test.",
            "The supplied north.csv is not exactly reproduced by its supplied map and current code.",
            "Source labels may conflict with the upstream request vocabulary; source pairs are retained.",
            "The map buffer is geometric and does not impose road, shoreline or land-use constraints.",
        ],
        "software": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "matplotlib", "pyproj", "shapely")
        }},
        "statistics": stats, "users": trace,
    }
    write_json(output / f"{scene}_generation.json", metadata)
    return {"scene": scene, "buildings": buildings, "tx": tx, "bounds": bounds,
            "positions": positions, "rows": rows, "statistics": stats}


def plot_scenarios(scenes: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(scenes), figsize=(6 * len(scenes), 6),
                             squeeze=False, constrained_layout=True)
    for ax, data in zip(axes[0], scenes):
        for building in data["buildings"]:
            xs, ys = zip(*building["nodes"])
            ax.fill(xs, ys, color="#e4e7eb", edgecolor="#89929c", linewidth=0.6)
        xs, ys, _ = zip(*data["positions"])
        scatter = ax.scatter(xs, ys, c=[r["CQI"] for r in data["rows"]],
                             cmap="viridis", vmin=1, vmax=15, s=45, edgecolors="white", linewidths=0.5)
        ax.scatter(*data["tx"][:2], marker="^", c="#d62728", s=100, label="BS", zorder=5)
        if len(data["rows"]) <= 40:
            for row in data["rows"]:
                ax.annotate(row["RX_ID"], (float(row["X"]), float(row["Y"])),
                            xytext=(4, 3), textcoords="offset points", fontsize=7)
        bounds = data["bounds"]
        stats = data["statistics"]
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="X (m)", ylabel="Y (m)",
               title=f'{data["scene"].title()} | {stats["users"]} users | CQI {stats["cqi_min"]}-{stats["cqi_max"]}')
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right")
    fig.colorbar(scatter, ax=list(axes[0]), label="CQI (upstream SNR mapping)", shrink=0.72)
    fig.savefig(output / "scenarios_cqi.png", dpi=180)
    plt.close(fig)


def user_count(value: str) -> int | str:
    if value == "source":
        return value
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("users must be a positive integer or 'source'") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("users must be a positive integer or 'source'")
    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--users", type=user_count, default=30, help="users per scene (default: 30); 'source' retains source counts")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path, default=ROOT / "data/scenarios", help="new output directory; existing directories are refused")
    args = parser.parse_args(argv)
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Duplicate scenarios are not allowed")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_model()
    # Publish the group only after every scene and preview succeeds.
    with tempfile.TemporaryDirectory(prefix=".scenario_generation_", dir=output.parent) as tmp:
        staging = Path(tmp) / "output"
        staging.mkdir()
        scenes = [generate_scene(model, scene, args.users, args.seed, staging) for scene in args.scenarios]
        plot_scenarios(scenes, staging)
        write_json(staging / "summary.json", {d["scene"]: d["statistics"] for d in scenes})
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
        staging.rename(output)
    for data in scenes:
        stats = data["statistics"]
        print(f'{data["scene"]}: {stats["users"]} users, labels={stats["label_counts"]}, '
              f'CQI={stats["cqi_min"]}-{stats["cqi_max"]}, mean={stats["cqi_mean"]:.2f}')
    print(f"Saved scenarios to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
