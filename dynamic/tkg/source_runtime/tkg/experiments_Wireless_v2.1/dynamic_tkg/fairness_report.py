"""Independent interval integration for paired cumulative-fairness reports.

Snapshots at t_i describe the service delivered over [t_i, t_i+1). The terminal
snapshot closes the final interval and never supplies extra traffic. Rates and
service validity are recomputed from allocations/current CQI/contracts rather
than trusting a planner's projections or graph traffic accumulators.
"""

from collections import defaultdict
from copy import deepcopy
import csv
import json
import math
from pathlib import Path

from .service import SLICE_LIMITS, evaluate_service


SCENARIOS = ("center", "north", "south")
SUMMARY_METRICS = (
    "effective_user_time_ratio", "degraded_user_seconds",
    "time_mean_effective_throughput_mbps", "time_mean_raw_throughput_mbps",
    "final_cumulative_jain_rate", "final_cumulative_jain_demand_normalized",
    "final_cumulative_jain_rate_eMBB", "final_cumulative_jain_rate_URLLC",
    "final_cumulative_jain_demand_normalized_eMBB",
    "final_cumulative_jain_demand_normalized_URLLC",
)


def _jain(values):
    denominator = len(values) * sum(value * value for value in values)
    return sum(values) ** 2 / denominator if denominator else 0.0


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty report: {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _observations(state):
    if "users" in state:
        return [{"RX_ID": str(uid), "CQI": user["cqi"]}
                for uid, user in state["users"].items()]
    observations = state.get("observations")
    if isinstance(observations, dict):
        return [{"RX_ID": str(uid), "CQI": row.get("CQI", row.get("cqi"))}
                for uid, row in observations.items()]
    if isinstance(observations, list):
        return observations
    raise ValueError("Snapshot has no current user CQI observations")


def _cohort_contracts(state):
    return {str(uid): {key: state["contracts"][str(uid)][key]
                      for key in ("slice_type", "min_rate_mbps", "max_rate_mbps")}
            for uid in state["allocations"]}


def _load_states(path):
    grouped = defaultdict(list)
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                state = json.loads(line)
                grouped[state["scenario"]].append(state)
    if set(grouped) != set(SCENARIOS):
        raise ValueError("Fairness comparison requires center, north and south")
    for scene, states in grouped.items():
        states.sort(key=lambda state: float(state["time_s"]))
        if len(states) < 2:
            raise ValueError(f"{scene}:need at least two timestamped snapshots")
        times = [float(state["time_s"]) for state in states]
        if (not all(math.isfinite(value) for value in times) or times[0] != 0
                or any(after <= before for before, after in zip(times, times[1:]))):
            raise ValueError(f"{scene}:timestamps must start at zero and strictly increase")
        if [state["frame_index"] for state in states] != list(range(len(states))):
            raise ValueError(f"{scene}:missing or duplicate frame indices")
        first_contracts = _cohort_contracts(states[0])
        if not first_contracts:
            raise ValueError(f"{scene}:empty admitted cohort")
        for state in states:
            if _cohort_contracts(state) != first_contracts:
                raise ValueError(f"{scene}:admitted identities or contracts changed")
            evaluated = evaluate_service(_observations(state), state["allocations"], state["contracts"])
            for key in ("effective_users", "admitted_users", "raw_throughput_mbps", "effective_throughput_mbps"):
                if key in state.get("metrics", {}) and not math.isclose(
                        float(state["metrics"][key]), float(evaluated["metrics"][key]),
                        rel_tol=1e-10, abs_tol=1e-7):
                    raise ValueError(f"{scene}/{state['frame_index']}:recomputed {key} differs from snapshot")
            state["_evaluated"] = evaluated
    return grouped


def _validate_pair(baseline, controlled):
    for scene in SCENARIOS:
        before, after = baseline[scene], controlled[scene]
        if [(s["frame_index"], s["time_s"]) for s in before] != [
                (s["frame_index"], s["time_s"]) for s in after]:
            raise ValueError(f"{scene}:unpaired timelines")
        if _cohort_contracts(before[0]) != _cohort_contracts(after[0]):
            raise ValueError(f"{scene}:unpaired admitted identities or contracts")
        if before[0]["allocations"] != after[0]["allocations"]:
            raise ValueError(f"{scene}:t=0 must preserve the same initial allocation")
        if after[-1]["allocations"] != after[-2]["allocations"]:
            raise ValueError(f"{scene}:terminal frame must not make a new allocation")
        for old, new in zip(before, after):
            old_cqi = {str(row.get("RX_ID", row.get("user_id"))): float(row.get("CQI", row.get("cqi")))
                       for row in _observations(old)}
            new_cqi = {str(row.get("RX_ID", row.get("user_id"))): float(row.get("CQI", row.get("cqi")))
                       for row in _observations(new)}
            if old_cqi != new_cqi:
                raise ValueError(f"{scene}/{old['frame_index']}:unpaired current CQI")
            if old["allocations"] != before[0]["allocations"]:
                raise ValueError(f"{scene}:no_agent baseline must retain its initial allocation")


def _integrate(method, scene, states):
    first = states[0]
    contracts = _cohort_contracts(first)
    ids = sorted(contracts)
    volumes = dict.fromkeys(ids, 0.0)
    effective_times = dict.fromkeys(ids, 0.0)
    deficit_volumes = dict.fromkeys(ids, 0.0)
    raw_mbit = effective_mbit = effective_user_seconds = 0.0
    rows = []
    for index, state in enumerate(states):
        time_s = float(state["time_s"])
        if index:
            previous = states[index - 1]
            dt = time_s - float(previous["time_s"])
            previous_users = previous["_evaluated"]["users"]
            for uid in ids:
                delivered = float(previous_users[uid]["rate_mbps"])
                valid = bool(previous_users[uid]["effective"])
                volumes[uid] += delivered * dt
                effective_times[uid] += int(valid) * dt
                deficit_volumes[uid] += max(float(contracts[uid]["min_rate_mbps"]) - delivered, 0.0) * dt
                raw_mbit += delivered * dt
                effective_mbit += delivered * dt if valid else 0.0
                effective_user_seconds += int(valid) * dt
        metrics = state["_evaluated"]["metrics"]
        raw = [volumes[uid] for uid in ids]
        normalized = [volumes[uid] / float(contracts[uid]["min_rate_mbps"]) for uid in ids]
        row = {
            "method": method, "scenario": scene, "frame_index": state["frame_index"],
            "time_s": time_s, "history_duration_s": time_s, "admitted_users": len(ids),
            "effective_users": metrics["effective_users"],
            "raw_throughput_mbps": metrics["raw_throughput_mbps"],
            "effective_throughput_mbps": metrics["effective_throughput_mbps"],
            "cumulative_traffic_mbit": raw_mbit,
            "cumulative_effective_traffic_mbit": effective_mbit,
            "cumulative_effective_user_seconds": effective_user_seconds,
            "cumulative_effective_user_time_ratio": effective_user_seconds / (len(ids) * time_s)
            if time_s else None,
            "cumulative_jain_rate": _jain(raw) if time_s else None,
            "cumulative_jain_demand_normalized": _jain(normalized) if time_s else None,
            "cumulative_time_mean_raw_throughput_mbps": raw_mbit / time_s if time_s else None,
            "cumulative_time_mean_effective_throughput_mbps": effective_mbit / time_s if time_s else None,
        }
        for name in SLICE_LIMITS:
            selected = [uid for uid in ids if contracts[uid]["slice_type"] == name]
            row[f"cumulative_jain_rate_{name}"] = _jain([volumes[uid] for uid in selected]) if time_s and selected else None
            row[f"cumulative_jain_demand_normalized_{name}"] = _jain([
                volumes[uid] / float(contracts[uid]["min_rate_mbps"]) for uid in selected
            ]) if time_s and selected else None
        rows.append(row)

    duration = float(states[-1]["time_s"])
    summary = {
        "method": method, "scenario": scene, "frames": len(states),
        "duration_s": duration, "admitted_users": len(ids),
        "admitted_user_ids": ids,
        "effective_users_min": min(state["_evaluated"]["metrics"]["effective_users"] for state in states),
        "effective_users_final": states[-1]["_evaluated"]["metrics"]["effective_users"],
        "effective_user_time_ratio": effective_user_seconds / (len(ids) * duration),
        "degraded_user_seconds": len(ids) * duration - effective_user_seconds,
        "time_mean_raw_throughput_mbps": raw_mbit / duration,
        "time_mean_effective_throughput_mbps": effective_mbit / duration,
        "final_cumulative_jain_rate": rows[-1]["cumulative_jain_rate"],
        "final_cumulative_jain_demand_normalized": rows[-1]["cumulative_jain_demand_normalized"],
    }
    for name in SLICE_LIMITS:
        for metric in ("rate", "demand_normalized"):
            summary[f"final_cumulative_jain_{metric}_{name}"] = rows[-1][f"cumulative_jain_{metric}_{name}"]
    users = [{
        "method": method, "scenario": scene, "user_id": uid,
        "slice_type": contracts[uid]["slice_type"], "duration_s": duration,
        "traffic_mbit": volumes[uid], "mean_rate_mbps": volumes[uid] / duration,
        "minimum_rate_mbps": contracts[uid]["min_rate_mbps"],
        "demand_normalized_mean_rate": volumes[uid] / duration / float(contracts[uid]["min_rate_mbps"]),
        "effective_time_s": effective_times[uid],
        "effective_time_ratio": effective_times[uid] / duration,
        "rate_deficit_mbit": deficit_volumes[uid],
    } for uid in ids]
    return rows, summary, users


def _plot(rows, method, output, controller):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    numeric = controller == "numeric_validation"
    control_label = "Fairness tool (no model)" if numeric else "TKG-DTN fair agent"
    curves = (
        ("effective_users", "QoS-valid users"),
        ("cumulative_jain_demand_normalized", "Cumulative Jain: rate / minimum"),
        ("cumulative_jain_rate_eMBB", "Cumulative Jain: eMBB"),
        ("cumulative_jain_rate_URLLC", "Cumulative Jain: URLLC"),
    )
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True, sharey="row", layout="constrained")
        fig.suptitle("Cumulative service fairness on identical dynamic scenes\n" +
                     ("Deterministic tool validation; zero model/API calls" if numeric else
                      "TKG-DTN agent; same-frame control; measured model consumption reported separately"),
                     fontsize=14)
        for column, scene in enumerate(SCENARIOS):
            axes[0, column].set_title(scene.capitalize(), fontweight="bold")
            for source, label, color, style in (("no_agent", "No agent", "#64748b", "--"),
                                                (method, control_label, "#15803d", "-")):
                selected = [row for row in rows if row["method"] == source and row["scenario"] == scene]
                times = [row["time_s"] for row in selected]
                for index, (metric, ylabel) in enumerate(curves):
                    values = [float("nan") if row[metric] is None else row[metric] for row in selected]
                    if index == 0:
                        axes[index, column].step(times, values, where="post", label=label,
                                                 color=color, linestyle=style, linewidth=1.9)
                    else:
                        axes[index, column].plot(times, values, label=label,
                                                 color=color, linestyle=style, linewidth=1.9)
                    axes[index, column].grid(alpha=.2)
                    axes[index, column].set_xlim(times[0], times[-1])
                    if column == 0:
                        axes[index, column].set_ylabel(ylabel)
                    if index:
                        axes[index, column].set_ylim(0, 1.02)
            axes[0, column].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0, column].legend(loc="best", fontsize=8)
            axes[-1, column].set_xlabel("Simulation time (s)")
        path = output / "fairness_comparison.png"
        fig.savefig(path, dpi=210, bbox_inches="tight")
        plt.close(fig)
    return path.name


def write_fairness_report(output, baseline_output, controller, usage=None, *, plots=True):
    """Write paired causal traffic metrics and return the complete JSON report.

    ``numeric_validation`` is explicitly a deterministic tool run with zero API
    calls, not an LLM experiment. Other controller labels require actual usage
    records for consumption claims; missing usage remains unknown.
    """
    output, baseline_output = Path(output), Path(baseline_output)
    if not isinstance(controller, str) or not controller.strip():
        raise ValueError("A nonempty controller label is required")
    numeric = controller == "numeric_validation"
    method = "fair_numeric_validation" if numeric else "tkg_dtn_fair"
    baseline = _load_states(baseline_output / "frames.jsonl")
    controlled = _load_states(output / "frames_post_control.jsonl")
    _validate_pair(baseline, controlled)
    rows, summaries, user_rows = [], [], []
    for label, grouped in (("no_agent", baseline), (method, controlled)):
        for scene in SCENARIOS:
            frame_rows, summary, users = _integrate(label, scene, grouped[scene])
            rows.extend(frame_rows)
            summaries.append(summary)
            user_rows.extend(users)
    comparisons = []
    for scene in SCENARIOS:
        before = next(row for row in summaries if row["scenario"] == scene and row["method"] == "no_agent")
        after = next(row for row in summaries if row["scenario"] == scene and row["method"] == method)
        result = {"scenario": scene, "controller": controller, "controlled_method": method,
                  "admitted_users": after["admitted_users"], "duration_s": after["duration_s"]}
        for metric in SUMMARY_METRICS:
            result[f"no_agent_{metric}"] = before[metric]
            result[f"{method}_{metric}"] = after[metric]
            result[f"delta_{metric}"] = (after[metric] - before[metric]
                                         if after[metric] is not None and before[metric] is not None else None)
        comparisons.append(result)
    consumption = deepcopy(usage)
    if numeric:
        total = (usage or {}).get("total", usage or {})
        if any(total.get(key, 0) for key in ("api_attempts", "llm_calls", "prompt_tokens", "completion_tokens", "provider_total_tokens")):
            raise ValueError("Numeric validation must not be labeled zero-API when usage is nonzero")
        if consumption is None:
            consumption = {"total": {"llm_calls": 0, "api_attempts": 0,
                                      "prompt_tokens": 0, "completion_tokens": 0, "provider_total_tokens": 0}}
    _write_csv(output / "fairness_metrics.csv", rows)
    _write_csv(output / "cumulative_user_traffic.csv", user_rows)
    _write_csv(output / "comparison.csv", comparisons)
    report = {
        "controller": controller, "controlled_method": method,
        "is_model_experiment": not numeric, "repeats": 1,
        "baseline_output": str(baseline_output.resolve()),
        "comparison": comparisons, "summaries": summaries, "model_usage": consumption,
        "time_weighting": "Forward hold over [t_i,t_i+1); t=0 has no historical traffic; final t=300 closes [290,300) and adds no extra interval.",
        "control_timing": "Identical initial allocation, no t=0 control, no terminal-frame control. Intermediate actions are applied synchronously in the same frame.",
        "fairness_definition": "Jain of each fixed admitted user's cumulative delivered traffic / common elapsed time; raw and minimum-demand-normalized variants. All admitted users, including below-SLA users, remain in the denominator.",
        "slice_interpretation": "Pools and service minima differ between eMBB and URLLC; report per-slice Jain alongside overall raw and normalized indices.",
        "independent_reconstruction": "Rates and effectiveness recomputed from allocated bandwidth, current CQI and contracts; integrate previous post-control rates. Planner projections and graph traffic totals are not used.",
        "interpretation": ("Deterministic numerical validation of history-aware tools; zero model/API calls; these gains are not measured LLM-agent results."
                           if numeric else "One paired history-aware TKG-DTN agent run; gains cannot be attributed solely to the LLM or graph; no repeated-run confidence interval."),
        "outputs": ["fairness_metrics.csv", "cumulative_user_traffic.csv", "comparison.csv", "comparison.json"],
    }
    if plots:
        report["outputs"].append(_plot(rows, method, output, controller))
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
