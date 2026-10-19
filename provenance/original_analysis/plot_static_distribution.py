"""Plot Figure 4 initial user positions on its exact building map and viewport.
Colors are recorded static-input SNR, with no propagation recalculation.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
EXP = PROJECT / "DTN-KG-transfer-package-20260909/DTN-KG-new-transfer-20260909/DTN-KG-new/experiments_Wireless_v2"
BATCH = EXP / "results/four_methods_5repeats_20261012_212809"
SCENES = ("center",)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    dynamic_manifest_path = HERE / "dynamic_figure_validation.json"
    dynamic_manifest = json.loads(dynamic_manifest_path.read_text())
    frames = dynamic_manifest["map_layout"]["coordinate_frames"]
    experiment = EXP.parent / "experiments_Wireless_v2.1"
    geometry_path = experiment / "generate_scenarios.py"
    spec = importlib.util.spec_from_file_location("figure4_geometry", geometry_path)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    model = geometry.load_model()
    sources = {}
    for path in (dynamic_manifest_path, HERE / "dynamic_mobility_data.csv",
                 geometry_path, experiment / "vendor/upstream/RayTracing_cqi.py"):
        sources[str(path.relative_to(PROJECT))] = sha(path)
    assert sha(HERE / "dynamic_mobility_data.csv") == dynamic_manifest["output_sha256"]["dynamic_mobility_data.csv"]
    with (HERE / "dynamic_mobility_data.csv").open(encoding="utf-8-sig") as stream:
        initial = {(r["scenario"], r["user_id"]): r for r in csv.DictReader(stream) if int(r["time_s"]) == 0}
    assert len(initial) == 90
    scene_data, data_rows, checks = {}, [], {}
    for scene in SCENES:
        csv_path = EXP / f"data/scenarios/{scene}.csv"
        with csv_path.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        source_by_id = {r["RX_ID"]: r for r in rows}
        assert len(source_by_id) == len(rows)
        selected = json.loads((BATCH / f"{scene}__tkg_dtn__r001/selected_users.json").read_text())
        rows = [source_by_id[u["user_id"]] for u in selected]
        assert len(rows) == 30
        by_id = {r["RX_ID"]: r for r in rows}
        assert len(by_id) == 30
        sources[str(csv_path.relative_to(PROJECT))] = sha(csv_path)
        for method in ("tkg_dtn", "wireless_kb", "prompt", "rule_upstream"):
            for repeat in range(1, 6):
                path = BATCH / f"{scene}__{method}__r{repeat:03d}/selected_users.json"
                selected = json.loads(path.read_text())
                assert [u["user_id"] for u in selected] == [r["RX_ID"] for r in rows]
                for user in selected:
                    row = by_id[user["user_id"]]
                    assert all(math.isclose(user[k.lower()], float(row[k]), abs_tol=1e-8) for k in ("X", "Y", "Z"))
                    assert user["cqi"] == int(row["CQI"])
                    assert user["request"] == row["User_Request"]
                    assert user["ground_truth"] == row["Request_Label"]
                sources[str(path.relative_to(PROJECT))] = sha(path)
        map_path = experiment / f"data/maps/HKUST_{scene.title()}.osm"
        buildings, tx, origin, _ = geometry.scene_geometry(model, scene)
        # The Figure 4 publisher also binds the equivalent v2 map copies.
        reference_map = EXP / f"data/maps/HKUST_{scene.title()}.osm"
        assert sha(map_path) == sha(reference_map)
        assert sha(reference_map) == dynamic_manifest["source_sha256"][str(reference_map.relative_to(PROJECT))]
        sources[str(map_path.relative_to(PROJECT))] = sha(map_path)
        assert np.allclose(tx[:2], frames[scene]["source_bs_xy_m"], rtol=0, atol=1e-9)
        for r in rows:
            point = initial[scene, r["RX_ID"]]
            assert math.isclose(float(r["X"]), float(point["x_m"]), abs_tol=1e-8)
            assert math.isclose(float(r["Y"]), float(point["y_m"]), abs_tol=1e-8)
            assert math.isclose(float(r["X"]) - tx[0], float(point["x_bs_m"]), abs_tol=1e-8)
            assert math.isclose(float(r["Y"]) - tx[1], float(point["y_bs_m"]), abs_tol=1e-8)
        scene_data[scene] = (rows, buildings, tx)
        checks[scene] = {"users": len(rows), "matched_runs": 20, "matches_figure4_initial_positions": True,
                        "bs_source_xyz_m": list(tx), "geographic_origin": list(origin),
                        "snr_range_db": [min(float(r["SNR_dB"]) for r in rows), max(float(r["SNR_dB"]) for r in rows)]}
        for r in rows:
            data_rows.append({"scenario": scene, "user_id": r["RX_ID"],
                "x_source_m": r["X"], "y_source_m": r["Y"],
                "x_bs_m": float(r["X"]) - tx[0], "y_bs_m": float(r["Y"]) - tx[1],
                "snr_db": r["SNR_dB"], "cqi": r["CQI"], "los": r["LOS"]})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(4.2, 3.7))
    axes = [fig.add_axes([0.14, 0.15, 0.67, 0.76])]
    snrs = [float(r["snr_db"]) for r in data_rows]
    norm = Normalize(5 * math.floor(min(snrs) / 5), 5 * math.ceil(max(snrs) / 5))
    for index, (ax, scene) in enumerate(zip(axes, SCENES)):
        rows, buildings, tx = scene_data[scene]
        origin = np.array(tx[:2])
        all_nodes = []
        for building in buildings:
            nodes = np.asarray(building["nodes"]) - origin
            all_nodes.extend(nodes.tolist())
            ax.add_patch(Polygon(nodes, facecolor="#E5E8EC", edgecolor="#B4BBC4", linewidth=0.5, zorder=1))
        xy = np.array([[float(r["X"]), float(r["Y"])] for r in rows]) - origin
        scatter = ax.scatter(xy[:,0], xy[:,1], c=[float(r["SNR_dB"]) for r in rows],
            cmap="viridis", norm=norm, s=26, edgecolors="white", linewidths=0.45, zorder=3)
        ax.scatter(0, 0, marker="*", s=75, color="#BE3434", edgecolors="white", linewidths=0.65, zorder=11)
        xmin, xmax, ymin, ymax = frames[scene]["display_limits_xy_m"]
        assert all(xmin < x < xmax and ymin < y < ymax for x, y in xy)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_box_aspect(1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("N")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xticks([t for t in ax.get_xticks() if xmin <= t <= xmax])
        ax.set_yticks([t for t in ax.get_yticks() if ymin <= t <= ymax])
        ax.grid(color="#D9DEE5", linewidth=0.5, alpha=0.45)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#ADB5BD")
            spine.set_linewidth(0.7)
        ax.tick_params(colors="#59636D", labelsize=8.5, length=3)
        ax.set_xlabel("Local X (m)", fontsize=9, labelpad=3, color="#44515E")
        ax.set_ylabel("Local Y (m)", fontsize=9, labelpad=3, color="#44515E")
        checks[scene]["display_limits_xy_m"] = [*ax.get_xlim(), *ax.get_ylim()]
        assert not ax.lines, "Static map must contain no trajectory lines"
    cax = fig.add_axes([0.835, 0.15, 0.032, 0.76])
    bar = fig.colorbar(scatter, cax=cax, orientation="vertical")
    bar.set_label("SNR (dB)", fontsize=9, labelpad=4)
    bar.ax.tick_params(labelsize=8, length=2)
    axes[0].legend(handles=[Line2D([], [], marker="*", color="none", markerfacecolor="#BE3434", markeredgecolor="white", markersize=8, label="Base station"),
                        Line2D([], [], marker="o", color="none", markerfacecolor="#26828E", markeredgecolor="white", markersize=5, label="User")],
               loc="lower left", borderaxespad=0.5, borderpad=0.4, labelspacing=0.35,
               handlelength=1.1, handletextpad=0.4, frameon=True, facecolor="white",
               edgecolor="#D9DEE5", framealpha=0.95, fontsize=8)
    fig.canvas.draw()
    boxes = [list(ax.get_window_extent().bounds) for ax in axes]
    assert abs(boxes[0][2] - boxes[0][3]) < 1e-8
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"static_user_distribution.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    with (HERE / "static_user_distribution_data.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data_rows[0]))
        writer.writeheader()
        writer.writerows(data_rows)
    manifest = {"status": "passed", "scenes": checks,
                "coordinate_frame": "Source local Cartesian coordinates translated by the map BS XY; identical translation for users and buildings.",
                "snr": "Recorded data/scenarios SNR_dB, not recalculated; color scale covers the 30 Center users.",
                "color_limits_db": [norm.vmin, norm.vmax],
                "reference_figure": "dynamic_mobility (Figure 4)",
                "same_map_bs_viewport_as_figure4": True,
                "layout": "single column; center only; right vertical colorbar; lower-left internal legend",
                "axis_boxes_pixels_at_figure_dpi": boxes,
                "source_sha256": sources, "script_sha256": sha(Path(__file__)),
                "output_sha256": {n: sha(HERE / n) for n in ("static_user_distribution.pdf", "static_user_distribution.png", "static_user_distribution_data.csv")}}
    (HERE / "static_distribution_validation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(checks, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
