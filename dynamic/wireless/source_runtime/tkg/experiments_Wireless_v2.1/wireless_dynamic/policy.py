"""WirelessAgent's resource-review workflow adapted to a moving fixed cohort.

Static intent recognition and slice selection have already happened. Therefore
each dynamic decision retains the original resource-review and network-evaluation
LLM stages, without rerunning admission or intent classification. The original
KB content/rules are reused as contextual evidence. The numeric tool plans from
current CQI, inherited assigned slices, and the same previously established
service agreements supplied to TKG-DTN, never traffic history. This is a
documented dynamic adaptation, not upstream dynamic code.

``Policy(llm, output).step(state)`` returns an approved candidate or a hold.
It never changes the environment. The runner owns asynchronous application,
freshness/resource validation, delay accounting, and the evaluation ledger.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid


V2_ROOT = Path(__file__).resolve().parents[2] / "experiments_Wireless_v2"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from wireless_exp.policies import load_legacy
from continuation.planner import optimize_service
from continuation.service import SLICE_LIMITS, rate_mbps


ADAPTATION_VERSION = "wireless_kb_current_snapshot_two_stage_v1"
SYSTEM_PROMPT = """You are the WirelessAgent network slicing management expert.
The static admission/intent workflow is complete. This dynamic decision concerns
the currently admitted users with their inherited, immutable assigned slices.
Review a current snapshot and a deterministic integer-bandwidth resource plan.
Only current CQI, current allocation, current location, unchanged requests,
fixed service agreements and static application knowledge are available. Each decision begins a fresh
conversation. Do not infer unobserved traffic, a past trajectory or future CQI.

Preserve every admitted user and assigned slice. No admission, removal, migration,
request rewriting, contract rewriting or numerical bandwidth inventions are allowed.
Current SLA targets are the previously established service agreements shared
with the other controller; an inherited assigned slice may violate its agreement.
Such mismatches must remain observable and cannot authorize migration. The original
knowledge rules provide application context; they cannot authorize reclassification.
The exact resource tool maximizes the number meeting those current SLA targets,
then current effective throughput, then raw throughput, then minimizes bandwidth
change. You may approve its complete feasible plan or hold the current allocation.
Do not deliberately reduce throughput, fairness, or responsiveness.

The plan is a proposal, not an applied allocation. A separate runtime applies an
approved plan when the measured inference time has elapsed and validates it then.
Keep analysis concise and grounded in the supplied current-state evidence.
"""


def _write_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _packed(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def snapshot_inputs(state: dict) -> dict:
    """Explicit allowlist: never copy service/metrics/provenance or history.

    The existing service agreements are shared with TKG-DTN. Only their slice
    requirement and rate bounds are read; raw request labels, traffic histories,
    episode records and static provenance are not inspected, including indirectly
    via the environment's existing service/metrics fields.
"""
    allocations, cqis, contracts, rows = {}, {}, {}, []
    for raw_uid, raw in sorted(state["allocations"].items(), key=lambda item: str(item[0])):
        uid = str(raw_uid)
        if uid in allocations:
            raise ValueError("Duplicate normalized user ID")
        name, bandwidth = raw["slice_type"], raw["bandwidth"]
        if name not in SLICE_LIMITS:
            raise ValueError("Unknown inherited assigned slice")
        if (isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, float))
                or not math.isfinite(bandwidth) or bandwidth != int(bandwidth)):
            raise ValueError("Current bandwidth must be a finite integer")
        user = state["users"][uid]
        cqi = user["cqi"]
        if (isinstance(cqi, bool) or not isinstance(cqi, (int, float))
                or not math.isfinite(cqi) or cqi != int(cqi) or not 1 <= cqi <= 15):
            raise ValueError("Current CQI must be an integer in 1..15")
        location = list(user.get("location", []))
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in location):
            raise ValueError("Current location must contain finite coordinates")
        limits = SLICE_LIMITS[name]
        allocation = {"user_id": uid, "slice_type": name, "bandwidth": int(bandwidth)}
        allocations[uid], cqis[uid] = allocation, int(cqi)
        agreement = state["contracts"][uid]
        contracts[uid] = {"slice_type": str(agreement["slice_type"]),
                          "min_rate_mbps": float(agreement["min_rate_mbps"]),
                          "max_rate_mbps": float(agreement["max_rate_mbps"])}
        rows.append({"user_id": uid, "assigned_slice": name, "cqi": int(cqi),
            "bandwidth_mhz": int(bandwidth), "rate_mbps": rate_mbps(bandwidth, cqi),
            "location": location, "request": str(user.get("request", "")),
            "required_slice": contracts[uid]["slice_type"],
            "contract_min_rate_mbps": contracts[uid]["min_rate_mbps"],
            "contract_max_rate_mbps": contracts[uid]["max_rate_mbps"]})
    if not allocations:
        raise ValueError("Snapshot requires an admitted cohort")
    return {"scenario": str(state["scenario"]), "frame_index": int(state["frame_index"]),
            "time_s": float(state["time_s"]), "revision": int(state["revision"]),
            "allocations": allocations, "current_cqi": cqis,
            "service_agreements": contracts, "admitted_users": rows}


def _parse_action(response: str) -> tuple[str, str]:
    content = response.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        content = fence.group(1)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Accept one JSON decision surrounded by explanatory prose; never guess
        # an action from keyword matching or replace invalid model output.
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("WirelessAgent evaluation did not return a JSON decision") from None
        result = json.loads(content[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("WirelessAgent decision must be a JSON object")
    action, reason = result.get("action"), result.get("reason")
    if action not in ("approve", "hold") or not isinstance(reason, str) or not reason.strip():
        raise ValueError("WirelessAgent decision requires action approve/hold and a reason")
    return action, reason.strip()


class Policy:
    """Two real LLM stages; no cross-frame messages, counters or traffic inputs."""

    version = ADAPTATION_VERSION

    def __init__(self, llm, output):
        self.llm = llm
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.legacy = load_legacy("agent_kb")
        self.knowledge = self.legacy.KNOWLEDGE_BASE_CONTENT
        self.knowledge_sha256 = hashlib.sha256(self.knowledge.encode("utf-8")).hexdigest()
        self.source_files = [Path(__file__), V2_ROOT / "wireless_exp/legacy/agent_kb.py",
                             V2_ROOT / "data/knowledge/Intent_Understand.txt"]

    def step(self, state: dict) -> dict:
        started = time.perf_counter()
        calls_before = self.llm.calls
        failures_before = self.llm.failures
        decision_id = uuid.uuid4().hex
        snapshot = snapshot_inputs(state)
        self.llm.user_id = f"{snapshot['scenario']}:frame{snapshot['frame_index']:04d}"
        identity = {"decision_id": decision_id, "scenario": snapshot["scenario"],
                    "frame_index": snapshot["frame_index"], "time_s": snapshot["time_s"],
                    "source_revision": snapshot["revision"]}
        try:
            # Same current-state numeric allocation philosophy as adapted
            # static WirelessAgent, extended to best-effort service after CQI
            # changes. Established service agreements match the TKG comparator.
            plan = optimize_service(snapshot["allocations"], snapshot["current_cqi"],
                                    snapshot["service_agreements"])
            model_snapshot = {key: snapshot[key] for key in
                              ("scenario", "frame_index", "time_s", "revision", "admitted_users")}
            for row in model_snapshot["admitted_users"]:
                recommendation, reasons = self.legacy.get_application_slice_type(row["request"])
                row["knowledge_rule_recommendation"] = recommendation
                row["knowledge_rule_reasons"] = reasons
            preview = {"resource_valid": plan["resource_valid"],
                       "current_service_objective": plan["objective"],
                       "changes": plan["changes"], "violations": plan["violations"],
                       "irrecoverable_user_ids": plan["irrecoverable_user_ids"],
                       "objective_priority": plan["evidence"]["priority"]}
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content":
                         "Static application knowledge from the existing WirelessAgent KB:\n" + self.knowledge
                         + "\nCurrent snapshot:\n" + _packed(model_snapshot)
                         + "\nPublic physical constraints:\n" + _packed(SLICE_LIMITS)
                         + "\nResource allocation calculation results:\n" + _packed(preview)
                         + "\nStage 1, resource review: review the current whole-cohort plan, "
                           "its bandwidth/rate limits and fixed membership. In at most 180 words "
                           "state whether it is appropriate and explain any unavoidable breaches. "
                           "This review has not changed the allocation."}]
            _write_jsonl(self.output / "policy_inputs.jsonl", identity | {
                "version": self.version, "snapshot": model_snapshot, "preview": preview,
                "knowledge_sha256": self.knowledge_sha256,
                "information_scope": "current_snapshot_and_static_knowledge_only",
                "planning_contract_source": "same_fixed_service_agreements_as_tkg"})
            review = self._call(identity, "resource_review", messages)
            messages.append({"role": "assistant", "content": review})
            messages.append({"role": "user", "content":
                "Stage 2, network evaluation: evaluate the proposed whole-network outcome "
                "using the same current snapshot and resource review. Preserve all admitted "
                "users and assigned slices. Approve a feasible exact plan when appropriate, "
                "or hold with a specific evidence-based reason; an invalid plan must be held. "
                "Return only JSON: {\"action\":\"approve\" or \"hold\",\"reason\":\"concise reason\"}. "
                "Approval only submits a candidate for later runtime validation."})
            evaluation = self._call(identity, "network_evaluation", messages)
            if self.llm.failures != failures_before:
                raise RuntimeError("API transport failed; WirelessAgent fallback is forbidden")
            action, reason = _parse_action(evaluation)
            if action == "approve" and not plan["resource_valid"]:
                raise ValueError("WirelessAgent approved an invalid resource plan")
            approved = action == "approve"
            result = identity | {"version": self.version,
                "status": "approved" if approved else "held", "reason": reason,
                "allocations": deepcopy(plan["allocations"] if approved else snapshot["allocations"]),
                "changes": deepcopy(plan["changes"] if approved else []),
                "objective": deepcopy(plan["objective"]),
                "resource_valid": plan["resource_valid"],
                "decision_wall_seconds": time.perf_counter() - started,
                "llm_calls": self.llm.calls - calls_before,
                "knowledge_sha256": self.knowledge_sha256,
                "information_scope": "current_snapshot_and_static_knowledge_only",
                "planning_contract_source": "same_fixed_service_agreements_as_tkg",
                "environment_mutated": False}
            _write_jsonl(self.output / "policy_decisions.jsonl", result)
            return result
        except Exception as exc:
            # Never manufacture a successful decision or change resources on an
            # API/parse failure. The runner must retain this incomplete status.
            _write_jsonl(self.output / "policy_decisions.jsonl", identity | {
                "status": "failed", "error_type": type(exc).__name__,
                "decision_wall_seconds": time.perf_counter() - started,
                "llm_calls": self.llm.calls - calls_before, "environment_mutated": False})
            raise

    def _call(self, identity, stage, messages):
        started = time.perf_counter()
        response = self.llm.complete(deepcopy(messages))
        _write_jsonl(self.output / "policy_events.jsonl", identity | {
            "stage": stage, "messages": messages, "response": response,
            "elapsed_seconds": time.perf_counter() - started})
        return response
