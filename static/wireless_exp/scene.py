"""Run the supplied simplified propagation model on the included OSM maps."""

import csv
import math
import random
from pathlib import Path

from . import ROOT
from .data import sha256
from .policies import load_legacy
from .runner import write_json


def generate_scene(scene: str, users: int, seed: int, output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if scene not in ("center", "north", "south") or users < 1:
        raise ValueError("Invalid scene or user count")
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{scene}.csv"
    if target.exists():
        raise ValueError(f"Refusing to overwrite existing scenario: {target}")
    random.seed(seed)
    np.random.seed(seed)
    module = load_legacy("propagation")
    map_path = ROOT / "data/maps" / f"HKUST_{scene.title()}.osm"
    buildings = module.parse_osm_buildings(str(map_path))
    buildings, _, origin = module.convert_to_cartesian(buildings)
    tallest = module.find_tallest_building(buildings)
    if tallest is None:
        raise ValueError("No buildings parsed")
    tx = module.place_tx(tallest)
    nodes = [node for b in buildings for node in b["nodes"]]
    min_x, max_x = min(n[0] for n in nodes), max(n[0] for n in nodes)
    min_y, max_y = min(n[1] for n in nodes), max(n[1] for n in nodes)
    buffer = max(max_x - min_x, max_y - min_y) * 0.1
    positions = module.generate_rx_positions(
        (min_x - buffer, max_x + buffer, min_y - buffer, max_y + buffer), users, buildings
    )
    if len(positions) != users:
        raise ValueError(f"Only placed {len(positions)} / {users} users")
    results = module.perform_ray_tracing(tx, positions, buildings)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "RX_ID", "X", "Y", "Z", "SNR_dB", "RX_Power_dBm",
            "CQI", "LOS", "User_Request", "Request_Label",
        ])
        writer.writeheader()
        for index, item in enumerate(results, 1):
            writer.writerow({
                "RX_ID": index, **{k: f"{v:.2f}" for k, v in zip(("X", "Y", "Z"), item["position"])},
                "SNR_dB": f'{item["snr_dB"]:.2f}', "RX_Power_dBm": f'{item["rx_power_dBm"]:.2f}',
                "CQI": item["cqi"], "LOS": int(item["has_los"]),
                "User_Request": item["user_request"], "Request_Label": item["request_label"],
            })
    write_json(output / f"{scene}_generation.json", {
        "scenario": scene, "seed": seed, "users": users, "map_sha256": sha256(map_path),
        "csv_sha256": sha256(target), "tx_position": tx, "geographic_origin": origin,
        "model": "supplied LoS/NLoS path-loss approximation, NOT six-ray multipath simulation",
        "frequency_hz": 2.4e9, "noise_bandwidth_hz": 20e6, "tx_power_dbm": 30,
        "noise_floor_dbm": -174 + 10 * math.log10(20e6) + 8,
    })
    fig, ax = plt.subplots(figsize=(8, 7))
    for building in buildings:
        xs, ys = zip(*building["nodes"])
        ax.fill(xs, ys, color="lightgray", edgecolor="gray", alpha=0.7)
    ax.scatter(tx[0], tx[1], c="red", marker="^", s=80, label="BS")
    scatter = ax.scatter(
        [p[0] for p in positions], [p[1] for p in positions],
        c=[item["cqi"] for item in results], cmap="viridis", vmin=1, vmax=15,
    )
    fig.colorbar(scatter, ax=ax, label="CQI")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title=f"{scene}, seed={seed}, users={users}")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"{scene}_map.png", dpi=160)
    plt.close(fig)
    return target
