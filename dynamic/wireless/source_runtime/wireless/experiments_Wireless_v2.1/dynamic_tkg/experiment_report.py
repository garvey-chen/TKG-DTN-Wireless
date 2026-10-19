"""Paired temporal service metrics and actual provider-reported model usage."""

from collections import Counter
from pathlib import Path
import json
import math

from run_dynamic_graph import SCENES, write_csv, write_json


MEAN_METRICS = (
    "effective_users", "raw_throughput_mbps", "effective_throughput_mbps",
    "jain_rate_all_admitted", "jain_demand_normalized_all_admitted",
    "mean_rate_satisfaction_clipped_all_admitted",
)


def summarize_metrics(rows):
    """Integrate [t_i,t_i+1); the final 300 s sample adds no duration."""
    methods = {r["method"] for r in rows}
    if methods != {"no_agent", "tkg_dtn"}:
        raise ValueError("Paired comparison requires exactly no_agent and tkg_dtn methods")
    if {r["scenario"] for r in rows} != set(SCENES):
        raise ValueError("Paired comparison requires exactly the three configured scenarios")
    summaries = []
    paired = {}
    for method in sorted(methods):
        for scene in SCENES:
            selected = sorted((r for r in rows if r["method"] == method and r["scenario"] == scene),
                              key=lambda r: float(r["time_s"]))
            if len(selected) < 2:
                raise ValueError(f"Incomplete timeline for {method}/{scene}")
            times = [float(r["time_s"]) for r in selected]
            if (any(not math.isfinite(t) for t in times) or times[0] != 0
                    or any(b <= a for a, b in zip(times, times[1:]))):
                raise ValueError("Require finite increasing timestamps beginning at zero")
            indices = [int(r["frame_index"]) for r in selected]
            if indices != list(range(len(selected))):
                raise ValueError("Frame indices must start at zero without missing frames")
            counts = {int(r["admitted_users"]) for r in selected}
            if len(counts) != 1:
                raise ValueError("Fixed cohort size changed")
            count = counts.pop()
            if count <= 0:
                raise ValueError("Admitted cohort must be positive")
            signature = (tuple(zip(indices, times)), count)
            if scene in paired and paired[scene] != signature:
                raise ValueError(f"Unpaired frame timeline or admitted cohort size for {scene}")
            paired[scene] = signature
            duration = times[-1] - times[0]
            result = {"method": method, "scenario": scene, "admitted_users": count,
                      "frames": len(selected), "duration_s": duration,
                      "effective_users_min": min(int(r["effective_users"]) for r in selected),
                      "effective_users_final": int(selected[-1]["effective_users"])}
            for metric in MEAN_METRICS:
                result["time_mean_" + metric] = sum(float(a[metric]) * (float(b["time_s"]) - float(a["time_s"]))
                                                       for a, b in zip(selected, selected[1:])) / duration
            result["effective_user_time_ratio"] = result["time_mean_effective_users"] / count
            result["degraded_user_seconds"] = sum(
                (count - int(a["effective_users"])) * (float(b["time_s"]) - float(a["time_s"]))
                for a, b in zip(selected, selected[1:]))
            summaries.append(result)
    return summaries


def usage_summary(output):
    path = output / "llm_calls.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    decisions_path = output / "decisions.jsonl"
    decisions = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines()] if decisions_path.exists() else []
    result = {}
    for scene in (*SCENES, "total"):
        selected = [r for r in records if scene == "total" or str(r.get("user_id", "")).startswith(scene + ":")]
        actions = [d for d in decisions if scene == "total" or d.get("scenario", d.get("scene")) == scene]
        responses = [r for r in selected if "response" in r]
        used = [r for r in responses if isinstance(r.get("usage"), dict)]
        # A response and error can describe the same truncated attempt; count
        # (call, attempt) pairs, not lines, for actual HTTP attempts.
        attempts = {(r["call"], r["attempt"]) for r in selected}
        wall = [float(d.get("decision_wall_seconds", 0)) for d in actions]
        result[scene] = {
            "llm_calls": len({r["call"] for r in selected}), "api_attempts": len(attempts),
            "returned_responses": len(responses), "error_records": sum("error" in r for r in selected),
            "prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) for r in used),
            "completion_tokens": sum(r["usage"].get("completion_tokens", 0) for r in used),
            "provider_total_tokens": sum(r["usage"].get("total_tokens", 0) for r in used),
            "prompt_cache_hit_tokens": sum(r["usage"].get("prompt_cache_hit_tokens", 0) for r in used),
            "prompt_cache_miss_tokens": sum(r["usage"].get("prompt_cache_miss_tokens", 0) for r in used),
            "usage_reported_for_all_responses": bool(responses) and all(
                "prompt_tokens" in r.get("usage", {}) and "completion_tokens" in r.get("usage", {}) for r in responses),
            "model_requested": sorted({r["model_requested"] for r in responses}),
            "model_returned": sorted({r["model_returned"] for r in responses if r.get("model_returned")}),
            "response_api_wall_seconds_including_prior_retries": sum(r.get("elapsed_seconds", 0) for r in responses),
            "decision_count": len(actions), "decision_status_counts": dict(Counter(d.get("status", "unknown") for d in actions)),
            "decision_wall_seconds": sum(wall),
            "mean_decision_wall_seconds": sum(wall) / len(wall) if wall else 0,
            "max_decision_wall_seconds": max(wall, default=0),
            "decisions_exceeding_10s": sum(t > 10 for t in wall),
        }
    result["accounting"] = {
        "source": "Actual logged API response usage; transport errors without response usage have unknown billed consumption.",
        "price_or_bill": None,
        "price_note": "No account bill or configured unit prices used. Token counts are consumption, not a currency invoice.",
        "baseline_llm_calls": 0, "baseline_prompt_tokens": 0, "baseline_completion_tokens": 0,
        "time_note": "Decision/API wall time is execution overhead, not radio latency or simulated outage duration.",
    }
    return result


def plot_comparison(rows, usage, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    curves = (("effective_users", "QoS-valid users"),
              ("effective_throughput_mbps", "Effective throughput (Mbps)"),
              ("jain_demand_normalized_all_admitted", "Jain fairness: rate / minimum"))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True, sharey="row", layout="constrained")
        fig.suptitle("One paired run: frozen allocation vs TKG-DTN agent\n"
                     "20 admitted users per scene; identical telemetry; agent curves after same-frame control", fontsize=15)
        for column, scene in enumerate(SCENES):
            axes[0, column].set_title(scene.capitalize(), fontweight="bold")
            for method, label, color, style in (("no_agent", "No agent", "#64748b", "--"),
                                                  ("tkg_dtn", "TKG-DTN agent", "#15803d", "-")):
                selected = sorted((r for r in rows if r["scenario"] == scene and r["method"] == method),
                                  key=lambda r: float(r["time_s"]))
                times = [float(r["time_s"]) for r in selected]
                for row, (key, ylabel) in enumerate(curves):
                    axis = axes[row, column]
                    axis.step(times, [float(r[key]) for r in selected], where="post", label=label,
                              color=color, linestyle=style, linewidth=2)
                    axis.grid(alpha=0.2)
                    axis.set_xlim(times[0], times[-1])
                    if column == 0:
                        axis.set_ylabel(ylabel)
            axes[0, column].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes[2, column].set_ylim(0, 1)
            axes[2, column].set_xlabel("Simulation time (s)")
            axes[0, column].legend(loc="best")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"agent_comparison.{suffix}", dpi=240, bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
        labels = [s.capitalize() for s in SCENES]
        axes[0].bar(labels, [usage[s]["llm_calls"] for s in SCENES], color="#2563eb")
        axes[0].set_ylabel("Actual model calls")
        prompt = [usage[s]["prompt_tokens"] / 1000 for s in SCENES]
        axes[1].bar(labels, prompt, label="Input", color="#2563eb")
        axes[1].bar(labels, [usage[s]["completion_tokens"] / 1000 for s in SCENES],
                    bottom=prompt, label="Output", color="#f59e0b")
        axes[1].set_ylabel("Provider-reported tokens (thousands)")
        axes[1].legend()
        axes[2].bar(labels, [usage[s]["decision_wall_seconds"] for s in SCENES], color="#15803d")
        axes[2].set_ylabel("Total decision wall time (s)")
        fig.suptitle("TKG-DTN execution consumption; no-agent baseline uses zero model calls", fontsize=13)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"model_consumption.{suffix}", dpi=240, bbox_inches="tight")
        plt.close(fig)


def write_report(rows, output, *, plots=True):
    summaries = summarize_metrics(rows)
    write_csv(output / "method_summary.csv", list(summaries[0]), summaries)
    comparisons = []
    for scene in SCENES:
        baseline = next(s for s in summaries if s["scenario"] == scene and s["method"] == "no_agent")
        agent = next(s for s in summaries if s["scenario"] == scene and s["method"] == "tkg_dtn")
        item = {"scenario": scene}
        for key in ("effective_users_min", "effective_user_time_ratio", "degraded_user_seconds",
                    *("time_mean_" + k for k in MEAN_METRICS)):
            item["no_agent_" + key] = baseline[key]
            item["tkg_dtn_" + key] = agent[key]
            item["delta_" + key] = agent[key] - baseline[key]
        comparisons.append(item)
    write_csv(output / "comparison.csv", list(comparisons[0]), comparisons)
    usage = usage_summary(output)
    write_json(output / "model_usage.json", usage)
    write_csv(output / "model_usage.csv", ["scenario", *usage["total"]],
              [{"scenario": s, **usage[s]} for s in (*SCENES, "total")])
    write_json(output / "comparison.json", {
        "repeats": 1, "comparison": comparisons,
        "time_weighting": "Forward hold over [0,300); last sample contributes zero extra duration.",
        "control_timing": "Synchronous same-frame action; model wall time is reported separately and not injected into simulation.",
        "interpretation": "End-to-end TKG-DTN with adapted exact resource tools versus frozen allocation. Gains cannot be attributed solely to the LLM or graph; one run has no confidence interval.",
    })
    if plots:
        plot_comparison(rows, usage, output)
    return comparisons, usage
