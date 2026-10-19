#!/usr/bin/env python3
"""Run one method/scene with real model calls and completion-triggered control.

Each worker imports the exact method-specific source archived with the paper.
The simulation processes every integer telemetry tick and fractional completion
event. Real full-policy latency advances radio time; host replay bookkeeping and
the physical commit guard are modeled as instantaneous and reported separately.
No future sample is ever passed to a policy. Existing runs are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
EXPERIMENT = WORKSPACE / "dynamic"
DEFAULT_INPUT = WORKSPACE / "data/mobility"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path, value):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jain(values):
    denominator = len(values) * sum(v * v for v in values)
    return sum(values) ** 2 / denominator if denominator else 0.0


def event_times(start, end):
    """Every unseen integer sample, followed by an exact fractional event."""
    if not (math.isfinite(start) and math.isfinite(end) and end >= start):
        raise ValueError("Invalid event time range")
    result = [float(t) for t in range(math.floor(start) + 1, math.floor(end) + 1)]
    if end > start and (not result or result[-1] != end):
        result.append(float(end))
    return result


def load_runtime(method):
    runtime_key = "tkg" if method == "tkg_dtn" else "wireless"
    runtime = HERE / "source_runtime" / runtime_key
    provenance = json.loads((HERE / "runtime_config.json").read_text())[runtime_key]
    for relative, metadata in provenance["files"].items():
        if sha(runtime / relative) != metadata["sha256"]:
            raise AssertionError("Paper runtime source changed: " + relative)
    sys.path.insert(0, str(runtime / "experiments_Wireless_v2"))
    sys.path.insert(0, str(runtime / "experiments_Wireless_v2.1"))
    return runtime, provenance


class Experiment:
    def __init__(self, args, runtime, provenance, settings):
        from continuation.handoff import load_handoff
        from run_static_terminal_dynamic import verify_inherited_state
        self.verify_inherited_state = verify_inherited_state
        self.args, self.output = args, args.output
        self.runtime, self.provenance, self.settings = runtime, provenance, settings
        self.scene, self.method = args.scene, args.method
        self.handoff = EXPERIMENT / "data" / ("wireless_static_terminal_handoff_20261015"
            if self.method == "wireless_agent" else "static_terminal_handoff_20261014")
        self.terminals, self.handoff_manifest = load_handoff(self.handoff)
        self.samples = defaultdict(list)
        for row in read_csv(args.input_dir / f"{self.scene}.csv"):
            self.samples[int(row["Time_s"])].append(row)
        if set(self.samples) != set(range(301)) or any(len(rows) != 30 for rows in self.samples.values()):
            raise AssertionError("Require all 30 users at every second from 0 to 300")
        self.meta = json.loads((args.input_dir / "generation.json").read_text())["scenarios"][self.scene]
        self.graph = self.llm = self.policy = None
        self.state, self.event_index = None, -1
        self.intervals, self.decisions, self.metrics = [], [], {}
        self.ledger = {str(uid): {"traffic_mbit": 0.0, "effective_served_mbit": 0.0,
            "effective_time_s": 0.0, "observed_duration_s": 0.0} for uid in range(1, 31)}
        self.scratch = tempfile.TemporaryDirectory(prefix=f"wireless_event_{self.method}_{self.scene}_")
        self.scratch_path = Path(self.scratch.name)
        if self.method == "tkg_dtn":
            from event_graph_adapter import EventFairnessGraph
            self.graph = EventFairnessGraph(self.scratch_path / "actual_graph.sqlite",
                static_terminal_dir=self.handoff, history_window_s=None, horizon_s=10.0)
        else:
            from run_wireless_dynamic import observe, apply_decision
            self.observe, self.apply_wireless = observe, apply_decision
        if settings:
            from wireless_exp.llm import LoggedLLM
            self.llm = LoggedLLM(settings, self.output / "llm_calls.jsonl")
            if self.method == "wireless_agent":
                from wireless_dynamic.workflow_policy import Policy
                self.policy = Policy(self.llm, self.output / "policy", resource_engine_mode="matched_snapshot_fair")
        self.sync(0.0)
        self.initial = deepcopy(self.state)
        self.verify_inherited_state(self.state, self.handoff, initial=True)

    def annotate(self, state, kind):
        value = deepcopy(state)
        value.update(method=self.method, telemetry_time_s=float(math.floor(value["time_s"])),
                     event_kind=kind, event_index=self.event_index)
        return value

    def record_state(self, kind):
        append_json(self.output / "states.jsonl", self.annotate(self.state, kind))
        t = float(self.state["time_s"])
        if t.is_integer():
            self.metrics[int(t)] = self.metric()

    def metric(self):
        t = float(self.state["time_s"])
        normalized = [self.ledger[uid]["traffic_mbit"] / self.state["contracts"][uid]["min_rate_mbps"]
                      for uid in self.state["allocations"]]
        return {"method": self.method, "scenario": self.scene, "time_s": t,
            "telemetry_time_s": math.floor(t), "revision": self.state["revision"],
            "observed_users": 30, "admitted_users": len(self.state["allocations"]),
            **{k: self.state["metrics"][k] for k in (
                "effective_users", "effective_throughput_mbps", "raw_throughput_mbps")},
            "cumulative_jain_demand_normalized_admitted": jain(normalized) if t else None,
            "cumulative_effective_user_time_fraction": sum(v["effective_time_s"] for v in self.ledger.values()) / (30*t) if t else None}

    def integrate_to(self, end):
        start = float(self.state["time_s"])
        duration = end - start
        if duration <= 0:
            raise AssertionError("Nonpositive service interval")
        for uid, user in self.state["users"].items():
            service = user["service"]
            rate, effective = service["rate_mbps"], bool(service["effective"])
            interval = {"method": self.method, "scenario": self.scene, "user_id": uid,
                "start_time_s": start, "end_time_s": end, "duration_s": duration,
                "source_revision": self.state["revision"], "telemetry_time_s": math.floor(start),
                "cqi": user["cqi"], "bandwidth_mhz": service["bandwidth_mhz"],
                "rate_mbps": rate, "traffic_mbit": rate * duration,
                "service_effective": effective, "effective_served_mbit": rate*duration if effective else 0.0,
                "effective_time_s": duration if effective else 0.0,
                "measured_packet_traffic": False}
            self.intervals.append(interval)
            for field in ("traffic_mbit", "effective_served_mbit", "effective_time_s"):
                self.ledger[uid][field] += interval[field]
            self.ledger[uid]["observed_duration_s"] += duration

    def sync(self, timestamp):
        if self.state is not None:
            if timestamp == self.state["time_s"]:
                return
            self.integrate_to(timestamp)
        self.event_index += 1
        sample_time = math.floor(timestamp)
        if self.graph:
            self.state = self.graph.sync_event(self.scene, timestamp, self.samples[sample_time],
                sample_time_s=sample_time,
                scene_info={k: self.meta[k] for k in ("tx_position_m", "geographic_origin_lat_lon", "map_sha256")})
        else:
            rows = [{k: v for k, v in r.items() if k not in ("Scenario", "Frame_Index", "Time_s")}
                    for r in self.samples[sample_time]]
            self.state = self.observe(self.scene, self.event_index, timestamp,
                rows, self.terminals[self.scene], self.state)
        self.verify_inherited_state(self.state, self.handoff, initial=timestamp == 0)
        self.record_state("initial" if timestamp == 0 else "telemetry" if timestamp.is_integer() else "completion_observation")

    def advance(self, target):
        for t in event_times(float(self.state["time_s"]), float(target)):
            self.sync(t)

    def infer(self):
        before = self.llm.statistics()
        candidate_graph, policy_result, error = None, None, None
        candidate_allocations = deepcopy(self.state["allocations"])
        started = time.perf_counter()
        try:
            if self.graph:
                from continuation.adaptive_agent import AdaptiveFairDynamicTKGPolicy
                path = self.scratch_path / f"candidate_{len(self.decisions):04d}.sqlite"
                candidate_graph = self.graph.fork_candidate(path)
                policy = AdaptiveFairDynamicTKGPolicy(candidate_graph, self.llm, self.output / "policy", max_iterations=8)
                trigger = {"type": "previous_decision_completed", "observation_interval_s": 1,
                    "execution": "This round uses the current snapshot. Explicit commit stages a private candidate; the physical controller revalidates and applies it only after the whole decision completes.",
                    "source_sample_time_s": math.floor(self.state["time_s"]),
                    "planning_horizon_s": 10, "planning_horizon_is_not_execution_period": True}
                policy_result = policy.step(self.scene, trigger)
                candidate_allocations = deepcopy(candidate_graph.current_state(self.scene)["allocations"])
            else:
                policy_result = self.policy.step(deepcopy(self.state))
                if policy_result.get("workflow_nodes") != ["initialize", "understand_intent", "allocate_slice_type", "allocate_resources", "evaluate_network"]:
                    raise AssertionError("WirelessAgent did not execute its original five-node workflow")
                candidate_allocations = deepcopy(policy_result["allocations"])
        except Exception as exc:
            error = str(exc).replace(self.settings["api_key"], "[REDACTED]")
            policy_result = {"status": "failed", "error_type": type(exc).__name__, "error": error}
        finally:
            if candidate_graph is not None:
                candidate_graph.close()
        elapsed = time.perf_counter() - started
        # Removing our ephemeral candidate is replay housekeeping, not inference.
        if self.graph:
            for p in self.scratch_path.glob("candidate_*"):
                p.unlink()
        after = self.llm.statistics()
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise AssertionError("Invalid real decision latency")
        source = float(self.state["time_s"])
        decision = {"decision_id": policy_result.get("decision_id", f"{self.method}_{self.scene}_{len(self.decisions)+1:04d}"),
            "method": self.method, "scenario": self.scene, "decision_index": len(self.decisions)+1,
            "source_time_s": source, "source_sample_time_s": math.floor(source),
            "source_revision": self.state["revision"], "source_frame_index": self.state["frame_index"],
            "completed_time_s": source + elapsed, "scheduled_apply_time_s": source + elapsed,
            "decision_wall_seconds": elapsed, "policy_reported_seconds": policy_result.get("decision_wall_seconds"),
            "policy_status": policy_result["status"], "policy_result": policy_result,
            "allocations": candidate_allocations, "allocation_before": deepcopy(self.state["allocations"]),
            "applied_time_s": None, "checked_time_s": None,
            **{k: after[k]-before[k] for k in after},
            "total_tokens": after["prompt_tokens"]-before["prompt_tokens"]+after["completion_tokens"]-before["completion_tokens"]}
        return decision

    def apply(self, decision):
        started = time.perf_counter()
        decision["checked_time_s"] = self.state["time_s"]
        decision["validation_sample_time_s"] = math.floor(self.state["time_s"])
        decision["staleness_s"] = self.state["time_s"] - decision["source_time_s"]
        if self.graph:
            decision["status"] = "held"
            if decision["policy_status"] not in ("reallocated", "held"):
                decision["status"] = "policy_failed"
            elif decision["policy_status"] == "reallocated" and decision["allocations"] != self.state["allocations"]:
                try:
                    provenance = {"controller": "agent", "agent_executed": True,
                        "decision_id": decision["decision_id"], "source_revision": decision["source_revision"],
                        "source_time_s": decision["source_time_s"], "completed_time_s": decision["completed_time_s"],
                        "policy_result": decision["policy_result"], "control_timing": "completion_triggered_revalidated_original_candidate"}
                    self.state = self.graph.commit_allocations(self.scene, decision["allocations"],
                        expected_revision=self.state["revision"], decision=provenance)
                    decision.update(status="reallocated", applied_time_s=self.state["time_s"], applied_revision=self.state["revision"])
                except ValueError as exc:
                    decision.update(status="rejected_stale_or_invalid", validation_failure=str(exc))
        else:
            self.state = self.apply_wireless(self.state, decision)
        decision["executor_guard_wall_seconds"] = time.perf_counter() - started
        self.state["phase"] = "post_control"
        decision["allocation_after"] = deepcopy(self.state["allocations"])
        self.verify_inherited_state(self.state, self.handoff)
        self.record_state("decision_completed")

    def run(self):
        if self.method == "frozen":
            self.advance(300.0)
            return
        # The inherited t=0 state remains untouched; first fresh 1 s sample starts control.
        self.advance(1.0)
        consecutive_failures = 0
        while self.state["time_s"] < 300:
            decision = self.infer()
            completed = decision["completed_time_s"]
            self.advance(min(completed, 300.0))
            if completed >= 300:
                decision.update(status="terminal_drop", allocation_after=deepcopy(self.state["allocations"]))
            else:
                self.apply(decision)
            self.decisions.append(decision)
            append_json(self.output / "decisions.jsonl", decision)
            write_json(self.output / "progress.json", {"method": self.method, "scenario": self.scene,
                "time_s": self.state["time_s"], "decisions": len(self.decisions),
                "last_status": decision["status"], "last_decision_seconds": decision["decision_wall_seconds"],
                "llm_statistics": self.llm.statistics()})
            print(f"{self.method} {self.scene} round={len(self.decisions)} t={self.state['time_s']:.3f}s "
                  f"latency={decision['decision_wall_seconds']:.3f}s {decision['status']} "
                  f"effective={self.state['metrics']['effective_users']}", flush=True)
            consecutive_failures = consecutive_failures + 1 if decision["policy_status"] == "failed" else 0
            if consecutive_failures >= 3 and completed < 300:
                raise RuntimeError("Three consecutive policy failures; preserving incomplete run rather than silently substituting another controller")

    def finalize(self, complete):
        write_csv(self.output / "traffic_intervals.csv", self.intervals)
        write_csv(self.output / "frame_metrics.csv", [self.metrics[t] for t in sorted(self.metrics)])
        write_json(self.output / "final_traffic_history.json", {"method": self.method, "scenario": self.scene,
            "duration_s": self.state["time_s"], "users": self.ledger,
            "evaluation_only_not_policy_input": self.method != "tkg_dtn"})
        if complete:
            assert set(self.metrics) == set(range(301))
            assert all(math.isclose(v["observed_duration_s"], 300, abs_tol=1e-9) for v in self.ledger.values())
        if self.graph:
            history = self.graph.traffic_history(self.scene)
            for uid, row in history["users"].items():
                for k in ("traffic_mbit", "effective_served_mbit", "effective_time_s", "observed_duration_s"):
                    assert math.isclose(row[k], self.ledger[uid][k], rel_tol=1e-11, abs_tol=1e-8), (uid,k)
            write_json(self.output / "graph_traffic_history.json", {k:v for k,v in history.items() if k != "intervals"})
            assert self.graph.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert not self.graph.db.execute("PRAGMA foreign_key_check").fetchall()
            self.graph.close()
            self.graph = None
            shutil.copyfile(self.scratch_path / "actual_graph.sqlite", self.output / "graph.sqlite")
        if self.llm:
            self.llm.session.close()
        self.scratch.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("tkg_dtn", "wireless_agent", "frozen"), required=True)
    parser.add_argument("--scene", choices=("center", "north", "south"), required=True)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--credential-config", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.method not in ['wireless_agent', 'frozen']:
        parser.error("Use dynamic/tkg for tkg_dtn; dynamic/wireless for wireless_agent or frozen")
    args.input_dir = args.input_dir.resolve()
    args.output = (args.output or HERE / "runs" / f"{args.method}_{args.scene}").resolve()
    runtime, provenance = load_runtime(args.method)
    from continuation.handoff import load_handoff
    from wireless_exp.settings import load_llm_settings, public_settings
    if args.method == "tkg_dtn":
        from event_graph_adapter import EventFairnessGraph
        from continuation.adaptive_agent import AdaptiveFairDynamicTKGPolicy
    else:
        from run_wireless_dynamic import observe, apply_decision
        from wireless_dynamic.workflow_policy import Policy
    settings = None
    if args.method != "frozen":
        # Credentials only are read from local configuration. Keep the paper's
        # transport/model/temperature unchanged, despite newer live defaults.
        public = dict(provenance["original_model_settings"], max_calls_per_run=1000)
        settings = load_llm_settings({"llm": public}, args.credential_config)
        settings.update(public)
    if args.preflight_only:
        for name in ("static_terminal_handoff_20261014", "wireless_static_terminal_handoff_20261015"):
            load_handoff(EXPERIMENT / "data" / name)
        print(json.dumps({"preflight": "passed", "method": args.method,
            "scene": args.scene, "model_settings": public_settings(settings) if settings else None,
            "runtime_source_files": len(provenance["files"]), "api_calls": 0}, ensure_ascii=False))
        return
    if args.output.exists():
        parser.error("Refusing to overwrite existing run: " + str(args.output))
    args.output.mkdir(parents=True)
    inputs = {name:sha(args.input_dir / name) for name in (
        f"{args.scene}.csv", "timeline.csv", "generation.json")}
    source = {str(p.relative_to(HERE)):sha(p) for p in [Path(__file__), HERE / "event_graph_adapter.py"]}
    record = {"status": "running", "method": args.method, "scenario": args.scene,
        "duration_s": 300, "sample_interval_s": 1, "observed_users": 30, "repeats": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir), "input_sha256": inputs,
        "source_sha256": source, "runtime_directory": str(runtime), "paper_source": provenance,
        "model_settings": public_settings(settings) if settings else None,
        "agent_executed": args.method != "frozen",
        "timing_mode": "completion_triggered_measured_full_policy_delay",
        "first_decision_time_s": 1 if settings else None,
        "max_inflight_per_method_scene": 1, "fixed_control_interval_s": None,
        "planning_horizon_s": 10 if args.method == "tkg_dtn" else None,
        "planning_horizon_semantics": "Unchanged paper optimizer projection, independent of observation and execution timing; no future latency supplied.",
        "timing_semantics": "Measured full policy duration includes snapshot preparation, actual LLM calls/retries, tool work and policy logs. Apply at exact completion if <300 after latest floor-second sample is ingested; immediately launch next round. Old allocation persists during computation. First decision at t1; t0 is exact static handoff.",
        "harness_timing": "Discrete-event simulation with real API inference. Host telemetry replay, external physical commit guard and output bookkeeping are modeled instantaneous. Executor guard wall time is logged separately; no physical-network realtime claim.",
        "telemetry_semantics": "Integer-second instantaneous samples; zero-order hold between samples; no future observations available to policy.",
        "terminal_semantics": "A round begun before300 may finish afterwards; all actual costs retained but its candidate is dropped and contributes no service after300.",
        "fixed_membership": "All30 observations; admitted27/30/29 and assigned slices inherited, wrong-slice users remain.",
        "information_scope": "Graph-backed completed traffic and current/past evidence" if args.method == "tkg_dtn" else "Current snapshot only; cumulative traffic is evaluator-only",
        "source_isolated": True}
    write_json(args.output / "run.json", record)
    experiment, started, complete = None, time.perf_counter(), False
    try:
        experiment = Experiment(args, runtime, provenance, settings)
        record.update(initial_admitted_user_ids=list(experiment.state["allocations"]),
                      static_terminal_dir=str(experiment.handoff), static_terminal_provenance=experiment.handoff_manifest)
        write_json(args.output / "run.json", record)
        experiment.run()
        complete = True
        record["status"] = "complete"
    except BaseException as exc:
        error = str(exc).replace(settings["api_key"], "[REDACTED]") if settings else str(exc)
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=error)
        print(json.dumps({"status": record["status"], "error_type":type(exc).__name__, "error":error}), flush=True)
    finally:
        if experiment:
            try:
                experiment.finalize(complete)
            except Exception as exc:
                record.update(status="failed", finalization_error_type=type(exc).__name__, finalization_error=str(exc))
            usage = experiment.llm.statistics() if experiment.llm else {k:0 for k in (
                "llm_calls", "api_attempts", "llm_failures", "prompt_tokens", "completion_tokens")}
            usage["total_tokens"] = usage["prompt_tokens"]+usage["completion_tokens"]
            record.update(model_usage=usage, decision_count=len(experiment.decisions),
                decision_status_counts=dict(Counter(d["status"] for d in experiment.decisions)),
                completed_simulation_time_s=experiment.state["time_s"], final_metrics=experiment.metric())
            write_json(args.output / "model_usage.json", usage)
        for relative, digest in inputs.items():
            if sha(args.input_dir / relative) != digest:
                record.update(status="failed", input_changed=relative)
        for relative, digest in source.items():
            if sha(HERE / relative) != digest:
                record.update(status="failed", runner_source_changed=relative)
        for relative, metadata in provenance["files"].items():
            if sha(runtime / relative) != metadata["sha256"]:
                record.update(status="failed", policy_source_changed=relative)
        record.update(elapsed_wall_seconds=time.perf_counter()-started,
                      ended_at_utc=datetime.now(timezone.utc).isoformat())
        record["output_sha256"] = {str(p.relative_to(args.output)):sha(p) for p in sorted(args.output.rglob("*"))
                                  if p.is_file() and p.name != "run.json"}
        write_json(args.output / "run.json", record)
    if record["status"] != "complete":
        raise SystemExit(1)
    print(f"COMPLETE {args.method} {args.scene}: {len(experiment.decisions)} decisions", flush=True)


if __name__ == "__main__":
    main()
