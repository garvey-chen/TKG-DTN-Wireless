"""Run the original WirelessAgent StateGraph for current-snapshot maintenance.

One existing user anchors each whole-cohort maintenance invocation. Selection
uses only current assigned-slice rate satisfaction, with a stable user-ID tie.
All five original workflow nodes execute. Model prose is not a new commit gate:
as in the original workflow, its resource tool updates a private working copy
before network evaluation. The external runner owns delayed physical actuation.

The resource engine is explicit. ``original_static`` reuses the static baseline's
all-minima-feasible allocator. ``matched_snapshot`` uses the same current-frame
best-effort resource tool and service agreements as the dynamic comparison,
without any historical traffic or the TKG agent/planning workflow.
``matched_snapshot_fair`` adds current-rate sum-log utility after effective
user count, retaining the same snapshot information and hard constraints.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT.parent / "experiments_Wireless_v2"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from wireless_exp.metrics import SLICE_LIMITS, rate_mbps
from wireless_exp.shared_resources import (
    optimize_fixed_users, commit_admission, normalized, allocation_map,
)


VERSION = "wireless_kb_original_stategraph_maintenance_v2"
EXPECTED_NODES = ["initialize", "understand_intent", "allocate_slice_type",
                  "allocate_resources", "evaluate_network"]
ENGINE_MODES = ("original_static", "matched_snapshot", "matched_snapshot_fair")
ENGINE_OBJECTIVES = {
    "original_static": ["satisfy_all_assigned_slice_minima_or_hold", "maximize_current_total_throughput"],
    "matched_snapshot": ["maximize_current_effective_users", "maximize_current_effective_throughput",
                         "maximize_raw_throughput", "minimize_bandwidth_change"],
    "matched_snapshot_fair": ["maximize_current_effective_users", "maximize_current_proportional_fair_log_utility",
                              "maximize_current_effective_throughput", "minimize_bandwidth_change"],
}


def _write(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _id_key(uid):
    return (0, int(uid)) if str(uid).isdigit() else (1, str(uid))


def snapshot_inputs(state, resource_engine_mode="original_static"):
    """Allowlist current observations; do not read evaluator history/provenance."""
    if resource_engine_mode not in ENGINE_MODES:
        raise ValueError("Unknown resource engine mode")
    users, allocations, agreements = {}, {}, {}
    for raw_uid, raw in state["allocations"].items():
        uid = str(raw_uid)
        if uid in allocations:
            raise ValueError("Duplicate normalized admitted user ID")
        if str(raw.get("user_id", uid)) != uid:
            raise ValueError("Allocation user ID mismatch")
        name = raw["slice_type"]
        if name not in SLICE_LIMITS:
            raise ValueError("Unknown assigned slice")
        user = state["users"][uid]
        cqi, bandwidth = user["cqi"], raw["bandwidth"]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value != int(value) for value in (cqi, bandwidth)):
            raise ValueError("CQI and bandwidth must be finite integers")
        if not 1 <= cqi <= 15:
            raise ValueError("CQI must be in 1..15")
        location = list(user.get("location", []))
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in location):
            raise ValueError("Current location must contain finite coordinates")
        users[uid] = {"user_id": uid, "cqi": int(cqi), "location": location,
                      "request": str(user["request"])}
        allocations[uid] = {"user_id": uid, "slice_type": name, "bandwidth": int(bandwidth)}
        limits = SLICE_LIMITS[name]
        if resource_engine_mode != "original_static":
            contract = state["contracts"][uid]
            agreements[uid] = {"slice_type": str(contract["slice_type"]),
                               "min_rate_mbps": float(contract["min_rate_mbps"]),
                               "max_rate_mbps": float(contract["max_rate_mbps"])}
        else:
            agreements[uid] = {"slice_type": name, "min_rate_mbps": limits["rate"][0],
                               "max_rate_mbps": limits["rate"][1]}
    if not users:
        raise ValueError("Maintenance requires an existing admitted cohort")
    return {"scenario": str(state["scenario"]), "frame_index": int(state["frame_index"]),
            "time_s": float(state["time_s"]), "revision": int(state["revision"]),
            "users": users, "allocations": allocations, "service_agreements": agreements}


def select_maintenance_user(snapshot):
    return min(snapshot["allocations"], key=lambda uid: (
        rate_mbps(snapshot["allocations"][uid]["bandwidth"], snapshot["users"][uid]["cqi"])
        / SLICE_LIMITS[snapshot["allocations"][uid]["slice_type"]]["rate"][0],
        _id_key(uid)))


def _public_allocations(state):
    return {str(user["user_id"]): {"user_id": str(user["user_id"]), "slice_type": name,
                                  "bandwidth": int(user["bandwidth"])}
            for name, limits in SLICE_LIMITS.items() for user in state[limits["key"]]["users"]}


class Policy:
    version = VERSION

    def __init__(self, llm, output, resource_engine_mode="original_static"):
        if resource_engine_mode not in ENGINE_MODES:
            raise ValueError("Unknown resource engine mode")
        self.llm, self.output = llm, Path(output)
        self.resource_engine_mode = resource_engine_mode
        self.output.mkdir(parents=True, exist_ok=True)
        path = Path(__file__).with_name("workflow_agent_kb.py")
        spec = importlib.util.spec_from_file_location("wireless_maintenance_" + uuid.uuid4().hex, path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module.llm = llm
        original = V2 / "wireless_exp/legacy/agent_kb.py"
        if hashlib.sha256(original.read_bytes()).hexdigest() != self.module.ORIGINAL_STATIC_SHA256:
            raise ValueError("Original static workflow changed; maintenance fork must be reviewed")
        self.knowledge_sha256 = hashlib.sha256(self.module.KNOWLEDGE_BASE_CONTENT.encode()).hexdigest()
        self.source_files = [Path(__file__), path, original,
                             V2 / "wireless_exp/shared_resources.py", V2 / "wireless_exp/tkg/resources.py",
                             V2 / "data/knowledge/Intent_Understand.txt"]
        self.last_snapshot = None  # Cleared after each invocation; never a cross-frame input.

    def _legacy_state(self, snapshot):
        state = deepcopy(self.module.INITIAL_NETWORK_STATE)
        for name, limits in SLICE_LIMITS.items():
            part = state[limits["key"]]
            part["users"] = [
                {"user_id": uid, "cqi": snapshot["users"][uid]["cqi"],
                 "bandwidth": allocation["bandwidth"],
                 "rate": rate_mbps(allocation["bandwidth"], snapshot["users"][uid]["cqi"]),
                 "latency": limits["latency"][0]}
                for uid, allocation in snapshot["allocations"].items() if allocation["slice_type"] == name]
            part["resource_usage"] = sum(user["bandwidth"] for user in part["users"])
            part["utilization_rate"] = f"{100 * part['resource_usage'] / limits['capacity']:.2f}%"
        state["total_users"] = len(snapshot["allocations"])
        state["timestamp"] = f"simulation_t={snapshot['time_s']:g}s"
        return state

    def _plan_existing(self, state, uid, cqi, slice_type, latency):
        snapshot = self.last_snapshot
        current = _public_allocations(state)
        if (uid not in current or current[uid]["slice_type"] != slice_type
                or snapshot["users"][uid]["cqi"] != cqi):
            raise ValueError("Maintenance target must be an existing user on its inherited slice")
        candidate = None
        if self.resource_engine_mode == "original_static":
            numeric = optimize_fixed_users(state)
            feasible, violations = numeric["feasible"], numeric["violations"]
            if feasible:
                candidate = numeric["candidate_state"]
            degraded = []
        else:
            # Both matched modes use only this frame; no cumulative service
            # counters, historical graph or future telemetry enter the planner.
            from continuation.planner import optimize_service
            numeric = optimize_service(current, {key: user["cqi"] for key, user in snapshot["users"].items()},
                                       snapshot["service_agreements"],
                                       objective_mode=("instantaneous_fair" if self.resource_engine_mode
                                                       == "matched_snapshot_fair" else "throughput"))
            feasible, violations = numeric["resource_valid"], numeric["violations"]
            degraded = numeric["degraded_user_ids"]
            if feasible:
                candidate = deepcopy(state)
                for limits in SLICE_LIMITS.values():
                    for user in candidate[limits["key"]]["users"]:
                        user["bandwidth"] = numeric["allocations"][str(user["user_id"])]["bandwidth"]
                        user["rate"] = rate_mbps(user["bandwidth"], user["cqi"])
                candidate = normalized(candidate)
        limits = SLICE_LIMITS[slice_type]
        plan = {"feasible": feasible, "user_id": uid, "slice_type": slice_type,
                "base_state": normalized(state), "resource_engine": self.resource_engine_mode,
                "objective_priority": ENGINE_OBJECTIVES[self.resource_engine_mode],
                "numeric_objective": numeric.get("objective"),
                "violations": violations, "degraded_user_ids": degraded,
                "certificate": {"available_before_adjustment_mhz":
                                limits["capacity"] - state[limits["key"]]["resource_usage"]},
                "adjustments": [], "target_rate_valid": False,
                "rate_requirements": (str(snapshot["service_agreements"][uid]["min_rate_mbps"])
                    + "-" + str(snapshot["service_agreements"][uid]["max_rate_mbps"])
                    + " Mbps; required slice " + snapshot["service_agreements"][uid]["slice_type"])}
        if not feasible:
            return plan
        if (set(_public_allocations(candidate)) != set(current)
                or any(_public_allocations(candidate)[key]["slice_type"] != current[key]["slice_type"] for key in current)):
            raise ValueError("Maintenance tool altered fixed membership or slices")
        old, new = allocation_map(state), allocation_map(candidate)
        plan["adjustments"] = [
            {"user_id": key, "slice_type": old[key]["slice_type"],
             "old_bandwidth": old[key]["bandwidth"], "new_bandwidth": new[key]["bandwidth"],
             "old_rate": old[key]["rate"], "new_rate": new[key]["rate"],
             "bandwidth_change": new[key]["bandwidth"] - old[key]["bandwidth"],
             "bandwidth_reduction": old[key]["bandwidth"] - new[key]["bandwidth"]}
            for key in current if old[key]["bandwidth"] != new[key]["bandwidth"]]
        allocation = next(user for user in candidate[limits["key"]]["users"] if str(user["user_id"]) == uid)
        contract = snapshot["service_agreements"][uid]
        plan.update(candidate_state=candidate, allocation=deepcopy(allocation),
            target_rate_valid=(slice_type == contract["slice_type"]
                and contract["min_rate_mbps"] - 0.011 <= allocation["rate"] <= contract["max_rate_mbps"] + 0.011))
        return plan

    def _commit_existing(self, state, plan):
        if self.resource_engine_mode == "original_static":
            return commit_admission(state, plan)
        from continuation.planner import validate_allocations
        if not plan["feasible"] or normalized(state) != plan["base_state"]:
            raise ValueError("Invalid or stale private maintenance proposal")
        check = validate_allocations(_public_allocations(plan["candidate_state"]),
            {uid: user["cqi"] for uid, user in self.last_snapshot["users"].items()},
            self.last_snapshot["service_agreements"], _public_allocations(state))
        if not check["resource_valid"]:
            raise ValueError("Invalid maintenance proposal: " + str(check["violations"]))
        return deepcopy(plan["candidate_state"])

    def step(self, state):
        started = time.perf_counter()
        calls_before, failures_before = self.llm.calls, self.llm.failures
        snapshot = snapshot_inputs(state, self.resource_engine_mode)
        uid = select_maintenance_user(snapshot)
        identity = {"decision_id": uuid.uuid4().hex, "scenario": snapshot["scenario"],
                    "frame_index": snapshot["frame_index"], "time_s": snapshot["time_s"],
                    "source_revision": snapshot["revision"], "maintenance_user_id": uid,
                    "resource_engine_mode": self.resource_engine_mode, "version": self.version}
        self.llm.user_id = f"{snapshot['scenario']}:frame{snapshot['frame_index']:04d}"
        self.last_snapshot = snapshot
        semantics = ("Original static resource engine: retain every user and assigned slice; satisfy all "
                     "assigned-slice minima then maximize current throughput. If no such complete allocation "
                     "exists, hold all resources and report infeasibility."
                     if self.resource_engine_mode == "original_static" else
                     "Current-snapshot fairness resource tool: first maximize effective users; among equal-count "
                     "plans maximize sum(log(max(current_rate / contract_min_rate, 1e-9))) across all admitted "
                     "users, including users with a service breach; then maximize effective throughput and "
                     "minimize bandwidth changes. This is instantaneous proportional fairness, not cumulative "
                     "Jain maximization. Fixed service agreements and hard constraints remain unchanged. "
                     "Hard-valid plans may leave unavoidable service breaches. No traffic history is used."
                     if self.resource_engine_mode == "matched_snapshot_fair" else
                     "Shared current-snapshot resource tool: maximize current effective users, then effective "
                     "throughput, then raw throughput, then minimize bandwidth changes. Fixed service agreements "
                     "are shared across controllers. Hard-valid best-effort plans may leave unavoidable service "
                     "breaches; feasibility does not imply every minimum is satisfied. No traffic history is used.")
        self.module.reset_network_state()
        self.module.GLOBAL_NETWORK_STATE = self._legacy_state(snapshot)
        self.module.PREVIOUS_NETWORK_STATE = None
        self.module.MAINTENANCE_CONTEXT = {"assigned_slice": snapshot["allocations"][uid]["slice_type"],
            "resource_semantics": semantics, "plan_existing": self._plan_existing,
            "commit_existing": self._commit_existing, "tool_events": []}
        _write(self.output / "workflow_inputs.jsonl", identity | {"snapshot": snapshot,
            "selection_rule": "minimum_current_assigned_slice_rate_satisfaction_then_stable_user_id",
            "knowledge_sha256": self.knowledge_sha256,
            "original_static_sha256": self.module.ORIGINAL_STATIC_SHA256})
        initial = {"user_id": uid, "location": str(snapshot["users"][uid]["location"]),
                   "request": snapshot["users"][uid]["request"], "cqi": snapshot["users"][uid]["cqi"],
                   "history": [], "memory": {}, "step_count": 0, "current_step": "", "final_result": None}
        nodes, latest = [], initial
        try:
            graph = self.module.create_network_graph()
            with (self.output / "workflow_upstream.log").open("a", encoding="utf-8") as log, redirect_stdout(log):
                for update in graph.stream(initial, stream_mode="updates"):
                    for name, value in update.items():
                        if name.startswith("__"):
                            continue
                        nodes.append(name)
                        latest = value
                        _write(self.output / "workflow_nodes.jsonl", identity | {"node": name,
                            "node_index": len(nodes), "llm_calls_this_decision": self.llm.calls - calls_before,
                            "node_state": value})
            if nodes != EXPECTED_NODES:
                raise RuntimeError("Original five-node workflow did not finish: " + str(nodes))
            if self.llm.failures != failures_before:
                raise RuntimeError("Workflow API error; no hidden fallback is accepted")
            after = _public_allocations(self.module.get_current_network_state())
            if (set(after) != set(snapshot["allocations"])
                    or any(after[key]["slice_type"] != snapshot["allocations"][key]["slice_type"] for key in after)):
                raise ValueError("Workflow changed fixed membership or slices")
            failed = bool(latest["memory"].get("allocation_failed", False))
            plan = latest["memory"]["beamforming_result"]["resource_plan"]
            if failed and after != snapshot["allocations"]:
                raise AssertionError("Infeasible original workflow must hold its entire allocation")
            result = identity | {"status": "held" if failed else "approved", "allocations": after,
                "changes": deepcopy(plan["adjustments"] if not failed else []),
                "allocation_failed": failed, "resource_valid": plan["feasible"],
                "resource_violations": plan["violations"],
                "degraded_user_ids": plan["degraded_user_ids"],
                "objective_priority": plan["objective_priority"],
                "numeric_objective": plan["numeric_objective"],
                "reason": "Original workflow resource tool held an infeasible plan" if failed else
                          "Original workflow resource tool committed a private maintenance proposal",
                "workflow_nodes": nodes, "llm_calls": self.llm.calls - calls_before,
                "decision_wall_seconds": time.perf_counter() - started,
                "information_scope": "current_snapshot_only; no traffic history or future observations",
                "environment_mutated": False, "private_workflow_state_updated": not failed,
                "model_output_is_commit_gate": False,
                "original_static_sha256": self.module.ORIGINAL_STATIC_SHA256}
            _write(self.output / "workflow_policy_decisions.jsonl", result)
            return result
        except Exception as exc:
            _write(self.output / "workflow_policy_decisions.jsonl", identity | {
                "status": "failed", "error_type": type(exc).__name__, "workflow_nodes": nodes,
                "decision_wall_seconds": time.perf_counter() - started,
                "llm_calls": self.llm.calls - calls_before, "environment_mutated": False})
            raise
        finally:
            for index, event in enumerate(self.module.MAINTENANCE_CONTEXT["tool_events"], 1):
                _write(self.output / "workflow_resource_actions.jsonl", identity | {"event_index": index, **event})
            self.module.MAINTENANCE_CONTEXT = None
            self.module.reset_network_state()
            self.last_snapshot = None
