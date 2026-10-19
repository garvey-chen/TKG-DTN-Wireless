#!/usr/bin/env python3
"""Rebuild manuscript figures from all five complete, fixed-input repeats.

The CSV contains all 1,800 request records in the 60 scenario/method runs.
Trajectories show pointwise repeat means +/- sample standard deviation.
Costs first average 90 requests within each repeat and method, then report
the mean +/- sample standard deviation across five repeat-level averages.
Repeats are reruns of fixed inputs, not independently sampled workloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np


BATCH = "four_methods_5repeats_20261012_212809"
EXPERIMENT = Path(
    "DTN-KG-transfer-package-20260909/DTN-KG-new-transfer-20260909/"
    "DTN-KG-new/experiments_Wireless_v2"
)
SCENARIOS = ("center", "north", "south")
METHODS = ("tkg_dtn", "wireless_kb", "prompt", "rule_upstream")
REPEATS = tuple(range(1, 6))
NAMES = {
    "tkg_dtn": "TKG-DTN", "wireless_kb": "WirelessAgent",
    "prompt": "Prompt-based", "rule_upstream": "Rule-based",
}
COLORS = {
    "tkg_dtn": "#0072B2", "wireless_kb": "#D55E00",
    "prompt": "#009E73", "rule_upstream": "#666666",
}
MARKERS = {"tkg_dtn": "o", "wireless_kb": "s", "prompt": "D", "rule_upstream": "^"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-8)


def load_and_validate(results):
    series, rows, audits, hashes = {}, [], [], {}
    fixed_inputs = {}
    for scenario in SCENARIOS:
        for method in METHODS:
          for repeat in REPEATS:
            run_dir = results / f"{scenario}__{method}__r{repeat:03d}"
            summary_path, steps_path = run_dir / "summary.json", run_dir / "steps.jsonl"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            steps = read_jsonl(steps_path)
            calls_path = run_dir / "llm_calls.jsonl"
            calls = read_jsonl(calls_path) if calls_path.exists() else []
            users_path = run_dir / "selected_users.json"
            users = json.loads(users_path.read_text(encoding="utf-8"))
            if scenario in fixed_inputs:
                assert users == fixed_inputs[scenario], (run_dir, "inputs differ")
            else:
                fixed_inputs[scenario] = users
            for path in (summary_path, steps_path, calls_path, users_path):
                if path.exists():
                    hashes[str(path.relative_to(results))] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert summary["status"] == "complete", run_dir
            assert summary["repeat"] == repeat and summary["users_processed"] == 30, run_dir
            assert len(steps) == 30 and [s["arrivals"] for s in steps] == list(range(1, 31)), run_dir
            stats = summary.get("llm_statistics", {})
            assert stats.get("llm_failures", 0) == 0, run_dir
            assert len(calls) == stats.get("llm_calls", 0), run_dir
            assert len(calls) == stats.get("api_attempts", 0), run_dir
            assert len({c["call"] for c in calls}) == len(calls), run_dir
            tokens_in = sum(c["usage"]["prompt_tokens"] for c in calls)
            tokens_out = sum(c["usage"]["completion_tokens"] for c in calls)
            assert tokens_in == stats.get("prompt_tokens", 0), run_dir
            assert tokens_out == stats.get("completion_tokens", 0), run_dir
            assert close(np.mean([s["step_seconds"] for s in steps]), summary["mean_decision_seconds"]), run_dir
            assert steps[-1]["valid_correct_users"] == summary["final"]["valid_correct_users"], run_dir
            assert close(steps[-1]["resource_feasible_throughput_mbps"], summary["final"]["resource_feasible_throughput_mbps"]), run_dir
            calls_by_user = defaultdict(list)
            for call in calls:
                assert call["usage"]["total_tokens"] == call["usage"]["prompt_tokens"] + call["usage"]["completion_tokens"]
                calls_by_user[str(call["user_id"])].append(call)
            previous_calls = 0
            for step in steps:
                allocations = step["allocations"]
                used = sum(a["bandwidth"] for a in allocations)
                capacity = sum(step["state"][key]["total_capacity"]
                               for key in ("embb_slice", "urllc_slice"))
                assert close(capacity, 120), run_dir
                assert close(used, step["bandwidth_used_mhz"]), run_dir
                assert close(100 * used / capacity, step["bandwidth_utilization_pct"]), run_dir
                assert close(100 * (1 - used / capacity), step["bandwidth_idle_pct"]), run_dir
                assert -1e-8 <= step["bandwidth_idle_pct"] <= 100 + 1e-8, run_dir
                checked_users = sum(bool(a["resource_feasible"]) and bool(a["correct_slice"]) for a in allocations)
                checked_throughput = sum(a["calculated_rate_mbps"] for a in allocations if a["resource_feasible"])
                assert checked_users == step["valid_correct_users"], (run_dir, step["arrivals"])
                assert close(checked_throughput, step["resource_feasible_throughput_mbps"]), (run_dir, step["arrivals"])
                request_calls = calls_by_user[str(step["user_id"])]
                assert len(request_calls) == step["llm_calls_cumulative"] - previous_calls, (run_dir, step["arrivals"])
                previous_calls = step["llm_calls_cumulative"]
                rows.append({
                    "batch": BATCH, "repeat": repeat, "scenario": scenario,
                    "method": method, "method_label": NAMES[method],
                    "arrivals": step["arrivals"], "user_id": step["user_id"],
                    "valid_correct_users": checked_users,
                    "resource_feasible_throughput_mbps": checked_throughput,
                    "model_calls_this_request": len(request_calls),
                    "input_tokens_this_request": sum(c["usage"]["prompt_tokens"] for c in request_calls),
                    "output_tokens_this_request": sum(c["usage"]["completion_tokens"] for c in request_calls),
                    "decision_seconds": step["step_seconds"],
                    "bandwidth_idle_pct": step["bandwidth_idle_pct"],
                })
            series[scenario, method, repeat] = steps
            audits.append({
                "run": run_dir.name, "processed_requests": len(steps),
                "model_calls": len(calls), "input_tokens": tokens_in,
                "output_tokens": tokens_out, "mean_decision_seconds": summary["mean_decision_seconds"],
                "final_valid_correct_users": steps[-1]["valid_correct_users"],
                "final_resource_feasible_throughput_mbps": steps[-1]["resource_feasible_throughput_mbps"],
                "summary_matches_steps_and_llm_log": True,
                "all_step_counts_and_throughputs_match_allocation_flags": True,
            })
    overlap = {}
    for scenario in SCENARIOS:
        overlap[scenario] = {}
        for repeat in REPEATS:
            a = series[scenario, "tkg_dtn", repeat]
            b = series[scenario, "wireless_kb", repeat]
            overlap[scenario][f"r{repeat:03d}"] = {
                "user_count_equal_steps": sum(x["valid_correct_users"] == y["valid_correct_users"] for x, y in zip(a, b)),
                "throughput_equal_steps": sum(close(x["resource_feasible_throughput_mbps"], y["resource_feasible_throughput_mbps"]) for x, y in zip(a, b)),
                "max_absolute_user_count_difference": max(abs(x["valid_correct_users"] - y["valid_correct_users"]) for x, y in zip(a, b)),
                "max_absolute_throughput_difference_mbps": max(abs(x["resource_feasible_throughput_mbps"] - y["resource_feasible_throughput_mbps"]) for x, y in zip(a, b)),
            }
    aggregated = {}
    for method in METHODS:
        repeat_costs = []
        for repeat in REPEATS:
            group = [r for r in rows if r["method"] == method and r["repeat"] == repeat]
            assert len(group) == 90
            repeat_costs.append({
                "repeat": repeat, "requests": len(group),
                "model_calls_total": sum(r["model_calls_this_request"] for r in group),
                "input_tokens_total": sum(r["input_tokens_this_request"] for r in group),
                "output_tokens_total": sum(r["output_tokens_this_request"] for r in group),
                "model_calls_per_request": np.mean([r["model_calls_this_request"] for r in group]).item(),
                "input_tokens_per_request": np.mean([r["input_tokens_this_request"] for r in group]).item(),
                "output_tokens_per_request": np.mean([r["output_tokens_this_request"] for r in group]).item(),
                "total_tokens_per_request": np.mean([r["input_tokens_this_request"] + r["output_tokens_this_request"] for r in group]).item(),
                "mean_decision_seconds": np.mean([r["decision_seconds"] for r in group]).item(),
            })
        aggregated[method] = {"requests": 450, "repeat_costs": repeat_costs}
        for metric in ("model_calls_per_request", "input_tokens_per_request", "output_tokens_per_request", "total_tokens_per_request", "mean_decision_seconds"):
            values = [r[metric] for r in repeat_costs]
            aggregated[method][metric] = np.mean(values).item()
            aggregated[method][metric + "_sample_sd"] = np.std(values, ddof=1).item()
    return series, rows, {"runs": audits, "overlap_checks": overlap,
                           "fixed_inputs_identical_across_methods_and_repeats": True,
                           "aggregate_costs": aggregated, "source_sha256": hashes}


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8C8C8C")
    ax.tick_params(axis="both", length=3, color="#8C8C8C")


def save_figure(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def service_figure(series, output):
    fig, axes = plt.subplots(3, 3, figsize=(10.7, 7.7), sharex=True, sharey="row")
    styles = {
        "tkg_dtn": {"linestyle": "-", "linewidth": 2.0, "markevery": (1, 6), "zorder": 4},
        "wireless_kb": {"linestyle": (0, (4.0, 4.0)), "linewidth": 1.9, "markevery": (4, 6), "zorder": 5},
        "prompt": {"linestyle": "-.", "linewidth": 1.4, "markevery": (2, 6), "zorder": 3},
        "rule_upstream": {"linestyle": ":", "linewidth": 1.5, "markevery": (0, 6), "zorder": 2},
    }
    metrics = ("valid_correct_users", "resource_feasible_throughput_mbps", "bandwidth_idle_pct")
    for col, scenario in enumerate(SCENARIOS):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            for method in ("rule_upstream", "prompt", "tkg_dtn", "wireless_kb"):
                values = np.array([[s[metric] for s in series[scenario, method, repeat]] for repeat in REPEATS])
                mean, sd = np.mean(values, axis=0), np.std(values, axis=0, ddof=1)
                arrivals = np.arange(1, 31)
                ax.fill_between(arrivals, mean - sd, mean + sd,
                                color=COLORS[method], alpha=0.14, linewidth=0)
                ax.plot(arrivals, mean,
                        color=COLORS[method], marker=MARKERS[method], markersize=4.4,
                        markerfacecolor="white", markeredgewidth=1.15, **styles[method])
            style_axis(ax)
            ax.set_xlim(0, 31)
            ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
            if row == 0:
                ax.set_title(f"({chr(97+col)}) {scenario.capitalize()}", pad=9, fontsize=11)
                ax.set_ylim(0, 31)
                ax.yaxis.set_major_locator(MultipleLocator(5))
            elif row == 1:
                ax.set_ylim(0, 2000)
                ax.yaxis.set_major_locator(MultipleLocator(500))
            else:
                ax.set_ylim(-2, 102)
                ax.yaxis.set_major_locator(MultipleLocator(20))
                ax.set_xlabel("User arrivals")
    axes[0, 0].set_ylabel("Correct and feasible users")
    axes[1, 0].set_ylabel("Resource-feasible throughput\n(Mbps)")
    axes[2, 0].set_ylabel("Bandwidth idle ratio (%)")
    legend = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m],
                     markerfacecolor="white", markersize=5,
                     linewidth=styles[m]["linewidth"], linestyle=styles[m]["linestyle"],
                     label=NAMES[m]) for m in METHODS]
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.52, 1.0), handlelength=3.1, columnspacing=2.1)
    fig.subplots_adjust(left=0.09, right=0.992, bottom=0.095, top=0.915, wspace=0.13, hspace=0.16)
    save_figure(fig, output / "service_trajectories")


def cost_figure(aggregate, output):
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.65))
    x = np.arange(4)
    colors = [COLORS[m] for m in METHODS]
    labels = ["TKG-DTN", "Wireless\nAgent", "Prompt-\nbased", "Rule-\nbased"]
    calls = np.array([aggregate[m]["model_calls_per_request"] for m in METHODS])
    calls_sd = np.array([aggregate[m]["model_calls_per_request_sample_sd"] for m in METHODS])
    token_in = np.array([aggregate[m]["input_tokens_per_request"] for m in METHODS]) / 1000
    token_out = np.array([aggregate[m]["output_tokens_per_request"] for m in METHODS]) / 1000
    latency = np.array([aggregate[m]["mean_decision_seconds"] for m in METHODS])
    tokens_sd = np.array([aggregate[m]["total_tokens_per_request_sample_sd"] for m in METHODS]) / 1000
    latency_sd = np.array([aggregate[m]["mean_decision_seconds_sample_sd"] for m in METHODS])
    values = (calls, token_in + token_out, latency)
    standard_deviations = (calls_sd, tokens_sd, latency_sd)
    axes[0].bar(x, calls, width=0.61, color=colors)
    axes[1].bar(x, token_in, width=0.61, color=colors)
    axes[1].bar(x, token_out, bottom=token_in, width=0.61, color=colors,
                alpha=0.48, edgecolor="#333333", linewidth=0.5, hatch="////")
    axes[2].bar(x, latency, width=0.61, color=colors)
    titles = ("(a) Model calls", "(b) Token use", "(c) Decision latency")
    ylabels = ("Calls / request", "Thousand tokens / request", "Seconds / request")
    for index, (ax, data, sd) in enumerate(zip(axes, values, standard_deviations)):
        ax.errorbar(x, data, yerr=sd, fmt="none", ecolor="#202020", elinewidth=1.05, capsize=3.5, capthick=1.05)
        style_axis(ax)
        ax.set_title(titles[index], fontsize=11, pad=10)
        ax.set_ylabel(ylabels[index])
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelsize=9.1)
        ax.set_xlim(-0.58, 3.58)
        maximum = max(data + sd)
        ax.set_ylim(0, maximum * 1.19)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=index == 0))
        for position, value in enumerate(data):
            label = "0" if value == 0 else f"{value:.3f}" if index == 2 and value < 1 else f"{value:.2f}"
            ax.text(position, value + sd[position] + maximum * 0.025, label, ha="center", va="bottom", fontsize=9.5)
    fig.legend(handles=[Patch(facecolor="#999999", label="Input tokens"),
                        Patch(facecolor="#DDDDDD", edgecolor="#333333", hatch="////", label="Output tokens")],
               loc="lower center", bbox_to_anchor=(0.525, 0.03), ncol=2, frameon=False, fontsize=9.3)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.27, top=0.88, wspace=0.37)
    save_figure(fig, output / "decision_costs")


def main():
    output = Path(__file__).resolve().parent
    project = output.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=project / "outputs/unpacked_static")
    parser.add_argument("--output", type=Path, default=output)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    series, rows, audit = load_and_validate(args.results)
    with (args.output / "plot_data.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    service_figure(series, args.output)
    cost_figure(audit["aggregate_costs"], args.output)
    audit.update({
        "batch": BATCH, "repeats": list(REPEATS), "runs": audit["runs"],
        "source_directory": str(args.results.resolve()),
        "csv_rows": len(rows), "run_count": 60,
        "cost_denominator_per_method_per_repeat": 90,
        "cost_denominator_per_method_all_repeats": 450,
        "error_bars": "Sample standard deviation (ddof=1) across five repeat-level averages; token error bars use total tokens.",
        "trajectory_bands": "Pointwise sample standard deviation (ddof=1) across five fixed-input repeats.",
        "repeat_interpretation": "Fixed-input reruns, not independent sampled workloads; no confidence interval or request-level standard error is claimed.",
        "curve_markers": True,
        "overlap_rendering": "Unshifted solid blue and dashed orange trajectories; markers alternate at different arrival indices.",
        "bandwidth_idle_definition": "100 * (1 - sum of all allocated bandwidth / 120 MHz); checked against per-step utilization and state capacities.",
        "throughput_definition": "Sum of calculated rates over resource_feasible allocations; slice correctness is not additionally required.",
        "user_count_definition": "Count of allocations with both resource_feasible and correct_slice flags true.",
        "latency_definition": "For each method and repeat, mean of all 90 step_seconds across three scenarios; then mean and sample SD across five repeat-level means.",
        "token_definition": "Sum of usage.prompt_tokens and usage.completion_tokens from model-call logs; full prompt tokens include cached input.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {n: hashlib.sha256((args.output / n).read_bytes()).hexdigest()
                          for n in ("service_trajectories.pdf", "service_trajectories.png", "decision_costs.pdf", "decision_costs.png", "plot_data.csv")},
        "verification_status": "passed",
    })
    (args.output / "figure_validation.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "runs": 60, "csv_rows": len(rows),
                      "aggregate_costs": audit["aggregate_costs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
