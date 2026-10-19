"""Publication-friendly summaries of the observed frozen-allocation baseline."""

import math
from pathlib import Path


def plot_monitoring(metrics: list[dict], output: Path) -> list[Path]:
    """Plot measured users, throughput and fairness; do not simulate actions.

    An event-time guide marks the first recorded QoS breach in each scene that
    actually has a breach. It is derived from the supplied metrics, not a fixed
    timestamp or a claim that an agent has restored service.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    scenes = ("center", "north", "south")
    by_scene = {scene: sorted((row for row in metrics if row["scenario"] == scene),
                             key=lambda row: float(row["time_s"])) for scene in scenes}
    if any(not rows for rows in by_scene.values()):
        raise ValueError("Monitoring plot requires center, north and south metrics")
    for scene, rows in by_scene.items():
        times = [float(row["time_s"]) for row in rows]
        if any(not math.isfinite(time) for time in times) or len(set(times)) != len(times):
            raise ValueError(f"Nonfinite or duplicate timestamps in {scene}")

    def values(rows, column):
        return [float(row[column]) if row[column] is not None else float("nan") for row in rows]

    all_rows = [row for rows in by_scene.values() for row in rows]
    counts = [float(row[column]) for row in all_rows for column in ("admitted_users", "effective_users")]
    count_low, count_high = max(0, math.floor(min(counts)) - 1), math.ceil(max(counts)) + 1
    peak_rate = max(float(row[column]) for row in all_rows
                    for column in ("raw_throughput_mbps", "effective_throughput_mbps"))
    time_min = min(float(row["time_s"]) for row in all_rows)
    time_max = max(float(row["time_s"]) for row in all_rows)

    style = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.fontsize": 9, "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 3, figsize=(15.5, 10), sharex=True,
                                 sharey="row", layout="constrained")
        fig.suptitle("Frozen-allocation monitoring baseline\n"
                     "Position-dependent CQI; initial allocations held fixed; no agent actions", fontsize=16)
        for column, scene in enumerate(scenes):
            rows = by_scene[scene]
            time = values(rows, "time_s")
            axes[0, column].set_title(scene.capitalize(), pad=10, fontweight="bold")
            top = axes[0, column]
            top.plot(time, values(rows, "admitted_users"), color="#64748b", linewidth=2,
                     linestyle="--", label="Admitted users", zorder=2)
            top.plot(time, values(rows, "effective_users"), color="#15803d", linewidth=2,
                     label="QoS-valid users", zorder=3)
            top.set_ylim(count_low, count_high)
            top.yaxis.set_major_locator(MaxNLocator(integer=True))
            failed = [row for row in rows if float(row["effective_users"]) < float(row["admitted_users"])]
            if failed:
                top.scatter(values(failed, "time_s"), values(failed, "effective_users"),
                            color="#b91c1c", s=14, zorder=4, label="QoS breach observed")

            middle = axes[1, column]
            middle.plot(time, values(rows, "raw_throughput_mbps"), color="#2563eb",
                        linewidth=2, linestyle="--", label="All admitted users")
            middle.plot(time, values(rows, "effective_throughput_mbps"), color="#15803d",
                        linewidth=2, label="QoS-valid users only")
            middle.set_ylim(0, max(1, peak_rate * 1.12))

            bottom = axes[2, column]
            bottom.plot(time, values(rows, "jain_rate_all_admitted"), color="#7c3aed",
                        linewidth=2, label="Jain: raw rate")
            bottom.plot(time, values(rows, "jain_demand_normalized_all_admitted"), color="#c2410c",
                        linewidth=2, linestyle="--", label="Jain: rate / minimum demand")
            bottom.set_ylim(0, 1.04)
            bottom.set_xlabel("Simulation time (s)")

            for axis in axes[:, column]:
                axis.set_xlim(time_min, time_max if time_max > time_min else time_min + 1)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
                axis.grid(True, alpha=0.2, linewidth=0.7)
                axis.set_axisbelow(True)
                if failed:
                    axis.axvline(float(failed[0]["time_s"]), color="#9ca3af", linewidth=1,
                                 linestyle=":", zorder=1)
            if failed:
                first = float(failed[0]["time_s"])
                fraction = (first - time_min) / max(time_max - time_min, 1)
                top.text(first, 0.96, f"First QoS breach: {first:g} s", fontsize=9, color="#6b7280",
                         transform=top.get_xaxis_transform(), va="top",
                         ha="right" if fraction > 0.65 else "left")
            # Repeat legends in each scene so exported panel crops stay readable.
            for axis in axes[:, column]:
                axis.legend(loc="lower left", framealpha=0.88, edgecolor="none")

        axes[0, 0].set_ylabel("Users (count)")
        axes[1, 0].set_ylabel("Throughput (Mbps)")
        axes[2, 0].set_ylabel("Jain fairness (all admitted users)")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        paths = [output / "monitoring_overview.png", output / "monitoring_overview.pdf"]
        for path in paths:
            fig.savefig(path, dpi=260, bbox_inches="tight")
        plt.close(fig)
    return paths
