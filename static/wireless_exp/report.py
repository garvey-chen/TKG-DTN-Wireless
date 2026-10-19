"""Generate figures from saved trial logs only; never from upstream plot arrays."""

import csv
import json
from collections import defaultdict
from pathlib import Path


def report(directory: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grouped = defaultdict(list)
    summaries = []
    checkpoints = []
    for path in sorted(directory.rglob("summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["status"] != "complete":
            continue
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        group = (item["scenario"], item["method"])
        with (path.parent / "steps.jsonl").open(encoding="utf-8") as handle:
            steps = [json.loads(line) for line in handle]
        grouped[group].append((manifest, steps))
        selected = set(manifest["config"].get("checkpoints", [1, 5, 10, 15, 20, 25, 30]))
        for step in steps:
            if step["arrivals"] in selected:
                checkpoints.append({
                    "scenario": item["scenario"], "method": item["method"], "repeat": item["repeat"],
                    **{k: v for k, v in step.items() if k not in ("state", "result", "allocations", "violations")},
                })
        summaries.append({
            "scenario": item["scenario"], "method": item["method"], "repeat": item["repeat"],
            **item["final"], "steps_with_audit_issues": item["steps_with_audit_issues"],
            "mean_utilization_pct": item["mean_utilization_over_all_processed_arrivals_pct"],
        })
    if not grouped:
        raise ValueError("No complete trials found; no performance chart generated.")
    out = directory / "figures"
    out.mkdir(exist_ok=True)
    with (directory / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if checkpoints:
        with (directory / "checkpoints.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(checkpoints[0]))
            writer.writeheader()
            writer.writerows(checkpoints)
    for scene in sorted({g[0] for g in grouped}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        metrics = [
            ("raw_calculated_throughput_mbps", "All allocations (including violations)", "Throughput (Mbps)"),
            ("resource_feasible_throughput_mbps", "Resource-feasible allocations", "Throughput (Mbps)"),
            ("bandwidth_idle_pct", "Unused bandwidth", "Idle bandwidth (%)"),
            ("valid_correct_users", "Resource-feasible users in correct slice", "Users"),
        ]
        for (s, method), runs in sorted(grouped.items()):
            if s != scene:
                continue
            if len({len(steps) for _, steps in runs}) != 1:
                raise ValueError(f"Cannot aggregate different user counts: {s}/{method}")
            # Repeated trials must share experimental settings except repeat seed/order.
            signatures = {
                json.dumps({
                    "source": m["scenario_sha256"], "code": m["code_sha256"],
                    "users": m["users_requested"], "order": m["arrival_order"],
                    "llm": m.get("llm"),
                }, sort_keys=True)
                for m, _ in runs
            }
            if len(signatures) != 1:
                raise ValueError(f"Incompatible runs in {s}/{method}; report them separately.")
            x = [step["arrivals"] for step in runs[0][1]]
            for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
                values = np.array([[step[metric] for step in steps] for _, steps in runs])
                avg = values.mean(axis=0)
                ax.plot(x, avg, label=f"{method} (n={len(runs)})", marker=".", markersize=3)
                if len(runs) > 1:
                    std = values.std(axis=0, ddof=1)
                    ax.fill_between(x, avg - std, avg + std, alpha=0.12)
                ax.set_xlabel("Arriving users")
                ax.set_ylabel(ylabel)
                ax.set_title(title, fontsize=10)
                ax.grid(alpha=0.25)
        for ax in axes.flat:
            ax.legend(fontsize=8)
        study = "static wireless comparison" if any(s == scene and m == "tkg_dtn" for s, m in grouped) else "reproduction of supplied code"
        fig.suptitle(f"{scene}: {study}; audit retained (bands = ±1 SD)")
        fig.tight_layout()
        fig.savefig(out / f"{scene}.png", dpi=180)
        fig.savefig(out / f"{scene}.pdf")
        plt.close(fig)
    return {"complete_trials": len(summaries), "figures": str(out)}
