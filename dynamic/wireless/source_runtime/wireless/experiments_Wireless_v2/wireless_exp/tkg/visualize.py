"""Standalone figures rendered from exported Neo4j facts, without LLM calls."""

import json
from collections import defaultdict
from pathlib import Path


def draw_graphs(directory: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    snapshot = json.loads((directory / "graph_snapshot.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in snapshot["nodes"]}
    positions = {"BaseStation": (0, 3), "Slice": (3, 3), "Knowledge": (6, 3),
                 "User": (0, 0), "Allocation": (3, 0), "SchemaType": (6, 0),
                 "Observation": (0, -3), "StateEvent": (3, -3)}
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    for kind, (x, y) in positions.items():
        ax.text(x, y, kind, ha="center", va="center", fontsize=12,
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#e2efff", "edgecolor": "#24619c"}, zorder=3)
    for edge in snapshot["edges"]:
        if edge["type"] != "SCHEMA_RELATION":
            continue
        a, b = nodes[edge["source"]]["name"], nodes[edge["target"]]["name"]
        name = edge["properties"]["name"]
        if a == b:
            x, y = positions[a]
            ax.text(x, y - 0.65, "FOLLOWS: previous event", ha="center", va="top", fontsize=9)
            continue
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        bend = 0.25 if name == "HAS_ALLOCATION" else -0.25 if name == "CURRENT_ALLOCATION" else 0
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                shrinkA=48, shrinkB=48, connectionstyle=f"arc3,rad={bend}", color="#60738a")
        ax.add_patch(arrow)
        offset = -0.55 if bend > 0 else 0.55 if bend < 0 else 0.15
        ax.text((x1 + x2) / 2 + (0.5 if x1 == x2 else 0), (y1 + y2) / 2 + offset,
                name, ha="center", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    ax.text(6, -0.6, "Every instance has\nINSTANCE_OF -> its SchemaType.\n\nSchemaType nodes and\nSCHEMA_RELATION edges\nare stored in Neo4j.", ha="center", va="top", fontsize=10)
    ax.set_xlim(-1.4, 7.5)
    ax.set_ylim(-4.2, 4.1)
    ax.set_title("Wireless TKG-DTN: materialized Schema graph", fontsize=15)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(directory / ("schema_graph." + suffix), dpi=180)
    plt.close(fig)

    current = defaultdict(list)
    for edge in snapshot["edges"]:
        if edge["type"] == "CURRENT_ALLOCATION":
            allocation = nodes[edge["target"]]
            current[allocation["slice_type"]].append((nodes[edge["source"]], allocation))
    height = max(6, max((len(v) for v in current.values()), default=0) * 0.47 + 2)
    fig, axes = plt.subplots(1, 2, figsize=(14, height))
    for ax, name in zip(axes, ("eMBB", "URLLC")):
        ax.axis("off")
        rows = sorted(current[name], key=lambda pair: pair[0]["user_id"])
        count = max(1, len(rows))
        s = nodes["slice:" + name]
        ax.text(4.5, count / 2, f'{name}\n{s["resource_usage"]}/{s["capacity"]} MHz', ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.7", "facecolor": "#d3f2e3", "edgecolor": "#237a56"}, zorder=3)
        for index, (user, allocation) in enumerate(rows):
            y = count - index - 0.5
            ax.text(0, y, f'User {user["user_id"]}\nCQI {user["cqi"]}', ha="center", va="center", fontsize=8,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "#e2efff", "edgecolor": "#24619c"}, zorder=3)
            ax.text(2.2, y, f'{allocation["bandwidth"]} MHz\n{allocation["rate"]:.1f} Mbps', ha="center", va="center", fontsize=8,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff0d3", "edgecolor": "#b88227"}, zorder=3)
            for start, end in (((0.4, y), (1.65, y)), ((2.75, y), (4.1, count / 2))):
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, color="#9aa7b3", linewidth=0.7))
        ax.set_xlim(-0.8, 5.6)
        ax.set_ylim(-0.6, count + 0.6)
        ax.set_title(f"User -> CURRENT_ALLOCATION -> Allocation -> ON_SLICE -> {name}", fontsize=9)
    known = sum(n["kind"] == "User" for n in nodes.values())
    admitted = sum(len(v) for v in current.values())
    fig.suptitle(f'Current instance graph, revision {snapshot["revision"]}\n{admitted} allocated / {known} observed users; historical versions are retained in graph_snapshot.json', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in ("png", "svg"):
        fig.savefig(directory / ("instance_graph." + suffix), dpi=180)
    plt.close(fig)
    return {"schema": str(directory / "schema_graph.png"), "instances": str(directory / "instance_graph.png")}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(draw_graphs(args.graph_directory.resolve()), ensure_ascii=False))
