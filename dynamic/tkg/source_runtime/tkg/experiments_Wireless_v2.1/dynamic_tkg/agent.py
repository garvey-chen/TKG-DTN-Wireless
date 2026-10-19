"""Dynamic service-assurance adapter for the existing TKG-DTN ReAct agent.

The original v2 agent's streaming loop, action parser and terminal-result logic
are inherited unchanged. Only its task prompt and graph/resource tools change.
No API transport is replaced: runners pass the existing ``LoggedLLM`` instance.
Resource previews are deterministic tools; only an explicit model-selected
commit changes allocations. Errors never trigger a hidden resource fallback.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
import uuid


V2_ROOT = Path(__file__).resolve().parents[2] / "experiments_Wireless_v2"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from wireless_exp.tkg.rag_system_agent import NetworkRAGAgent

from .planner import optimize_service
from .schema import RELATION_TYPES
from .service import SLICE_LIMITS, legal_bandwidths, rate_mbps


TOOL_HELP = """
get_decision_context(): Refresh this decision's current graph state, known service
  contracts, constraints and recent events. It is already prefetched initially.
get_history(user_id?: str, event_type?: str, through_time_s?: number, limit?: int):
  Read past/current graph events, including service degradation and recovery.
  No future time is permitted; limit defaults to 20 and is at most 100.
get_bandwidth_options(user_id: str): Enumerate the selected admitted user's exact
  QoS-feasible integer MHz choices using current CQI and their existing slice.
plan_reallocation(): Nonmutating exact resource preview for the whole fixed
  admitted cohort. Maximizes effective users, then effective throughput, without
  changing membership, slices, contracts or CQI. Returns a plan_id, revision,
  changes, predicted metrics and feasibility evidence; no nested model calls.
commit_reallocation(plan_id: str): Apply a fresh, hard-resource-valid changed
  plan atomically to this graph. This ends the current decision. A best-effort
  plan may leave explicitly reported service breaches when full QoS is impossible.
hold(reason: str): End this decision without changing resources. State a specific
  evidence-based reason, for example unchanged optimum or irrecoverable channel.
Any tool supports full_evidence=true to return its complete logged evidence.
"""


def _packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _write_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _event_view(event):
    keys = ("event_id", "event_type", "user_id", "time_s", "frame_index", "revision",
            "reason", "episode_id", "cause")
    value = {key: deepcopy(event[key]) for key in keys if key in event}
    # Show useful channel deltas without carrying full status objects each round.
    if event.get("event_type") == "CQI_CHANGED":
        for key in ("before", "after"):
            if key in event:
                value[key] = deepcopy(event[key])
    return value


def _model_view(name, result):
    """Keep executable candidates private and model observations compact."""
    value = deepcopy(result)
    if "error" in value:
        return value
    if name == "get_decision_context":
        state = value.pop("state")
        columns = ["user_id", "slice_type", "cqi", "bandwidth_mhz", "rate_mbps",
                   "min_rate_mbps", "max_rate_mbps", "effective", "violations"]
        rows = []
        for uid, allocation in state["allocations"].items():
            user, contract = state["users"][uid], state["contracts"][uid]
            service = user["service"]
            rows.append([uid, allocation["slice_type"], user["cqi"], allocation["bandwidth"],
                         service["rate_mbps"], contract["min_rate_mbps"],
                         contract["max_rate_mbps"], service["effective"], service["violations"]])
        value["state"] = {
            "scenario": state["scenario"], "frame_index": state["frame_index"],
            "time_s": state["time_s"], "revision": state["revision"],
            "phase": state.get("phase", "post_control"), "metrics": state["metrics"],
            "admitted_user_columns": columns, "admitted_users": rows,
            "active_episodes": list(state["active_episodes"].values()),
        }
        value["events"] = [_event_view(event) for event in value["events"]]
    elif name == "get_history":
        value["events"] = [_event_view(event) for event in value["events"]]
    elif name == "get_bandwidth_options":
        value.pop("rates_mbps", None)
    elif name == "plan_reallocation":
        # All changes and objective/feasibility certificates remain visible.
        # Larger dynamic-programming diagnostics remain in full tool evidence.
        evidence = value.get("evidence", {})
        value["evidence"] = {
            key: deepcopy(evidence[key])
            for key in ("algorithm", "priority", "scope") if key in evidence
        }
    return value


class DynamicWirelessTools:
    version = "dynamic_fixed_cohort_exact_service_v1"

    def __init__(self, graph, output):
        self.graph = graph
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.result = None
        self.plans = {}
        self.context_read = False
        self.scene = None
        self.tool_calls = 0
        self.model_tool_calls = 0

    def start(self, scene, trigger_reason):
        state = self.graph.current_state(scene)
        self.scene = scene
        self.frame_index = state["frame_index"]
        self.time_s = state["time_s"]
        self.started_revision = state["revision"]
        self.decision_id = uuid.uuid4().hex
        self.trigger_reason = str(trigger_reason)
        self.result, self.plans, self.context_read = None, {}, False
        self.tool_calls = self.model_tool_calls = 0

    def _current(self):
        if self.scene is None:
            raise ValueError("Start a scene decision before using tools")
        state = self.graph.current_state(self.scene)
        if state["frame_index"] != self.frame_index or state["time_s"] != self.time_s:
            raise ValueError("Decision frame is stale; start a new decision")
        if self.result is not None:
            raise ValueError("This decision already has a terminal result")
        return state

    def _fresh_context(self):
        if not self.context_read:
            raise ValueError("Read get_decision_context before planning or deciding")
        return self._current()

    def get_decision_context(self):
        state = self._current()
        context = self.graph.read_context(self.scene)
        if context["state"]["revision"] != state["revision"]:
            raise ValueError("Graph changed while reading context; refresh the context")
        self.context_read = True
        events = self.graph.history(self.scene, through_time_s=state["time_s"],
                                    through_revision=state["revision"])
        admitted = set(state["allocations"])
        relevant = [event for event in events
                    if event.get("user_id") in admitted
                    and event["event_type"] != "POSITION_CHANGED"]
        relevant.sort(key=lambda event: (event["revision"], event["event_id"]))
        return {
            "decision_id": self.decision_id, "run_id": self.graph.run_id,
            "backend": "sqlite_local_property_graph", "trigger_reason": self.trigger_reason,
            "state": state, "constraints": deepcopy(SLICE_LIMITS),
            "events": relevant[-20:], "matching_past_event_count": len(relevant),
            "objective": ["maximize effective users in the fixed admitted cohort",
                          "then maximize effective throughput", "report fairness without optimizing it"],
            "assessment_scope": "Known contracts; rate, integer bandwidth, slice and pool constraints. Latency is unmeasured.",
        }

    def get_history(self, user_id=None, event_type=None, through_time_s=None, limit=20):
        state = self._fresh_context()
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("History limit must be an integer in 1..100")
        until = state["time_s"] if through_time_s is None else through_time_s
        if isinstance(until, bool) or not isinstance(until, (float, int)) or not math.isfinite(until):
            raise ValueError("History time must be a finite number")
        if not 0 <= until <= state["time_s"]:
            raise ValueError("History may only read observed past/current times")
        if user_id is not None and str(user_id) not in state["users"]:
            raise ValueError("Unknown user in this scene")
        events = self.graph.history(self.scene, through_time_s=until, user_id=user_id,
                                    event_type=event_type, through_revision=state["revision"])
        events.sort(key=lambda event: (event["revision"], event["event_id"]))
        return {"scenario": self.scene, "through_time_s": until,
                "through_revision": state["revision"], "matching_event_count": len(events),
                "events": events[-limit:]}

    def get_bandwidth_options(self, user_id):
        state = self._fresh_context()
        uid = str(user_id)
        if uid not in state["allocations"]:
            raise ValueError("Bandwidth tools operate on the fixed admitted cohort only")
        allocation, user = state["allocations"][uid], state["users"][uid]
        name, cqi = allocation["slice_type"], user["cqi"]
        limits = SLICE_LIMITS[name]
        choices = legal_bandwidths(name, cqi)
        return {"user_id": uid, "slice_type": name, "cqi": cqi,
                "revision": state["revision"], "contract": deepcopy(state["contracts"][uid]),
                "current_bandwidth_mhz": allocation["bandwidth"],
                "bandwidth_bounds_mhz": limits["bandwidth"], "pool_capacity_mhz": limits["capacity"],
                "qos_feasible_bandwidths_mhz": choices,
                "maximum_rate_at_bandwidth_bound_mbps": rate_mbps(limits["bandwidth"][1], cqi),
                "rates_mbps": {str(b): rate_mbps(b, cqi) for b in choices},
                "scope": "Individual QoS feasibility; plan_reallocation checks joint pool feasibility."}

    def plan_reallocation(self):
        state = self._fresh_context()
        cqis = {uid: user["cqi"] for uid, user in state["users"].items()}
        planned = optimize_service(state["allocations"], cqis, state["contracts"])
        plan_id = uuid.uuid4().hex[:16]
        plan = {
            **deepcopy(planned), "plan_id": plan_id, "scenario": self.scene,
            "frame_index": self.frame_index, "time_s": self.time_s,
            "revision": state["revision"], "metrics_before": deepcopy(state["metrics"]),
            "changed": planned["allocations"] != state["allocations"],
        }
        self.plans[plan_id] = plan
        return {key: deepcopy(value) for key, value in plan.items() if key != "allocations"}

    def commit_reallocation(self, plan_id):
        state = self._fresh_context()
        if not isinstance(plan_id, str) or plan_id not in self.plans:
            raise ValueError("Unknown plan_id; request a resource preview first")
        plan = self.plans[plan_id]
        if (plan["scenario"], plan["frame_index"], plan["revision"]) != (
                self.scene, state["frame_index"], state["revision"]):
            raise ValueError("Stale plan revision; preview again from current context")
        if not plan["resource_valid"]:
            raise ValueError("Hard-resource-invalid plans cannot be committed")
        if not plan["changed"]:
            raise ValueError("Plan makes no allocation changes; use hold with a reason")
        decision = {
            "decision_id": self.decision_id, "plan_id": plan_id, "agent": "TKG-DTN_dynamic",
            "trigger_reason": self.trigger_reason, "tool_version": self.version,
            "changes": deepcopy(plan["changes"]), "based_on_revision": state["revision"],
        }
        post = self.graph.commit_allocations(
            self.scene, plan["allocations"], expected_revision=plan["revision"], decision=decision)
        self.result = {
            **decision, "status": "reallocated", "scenario": self.scene,
            "frame_index": self.frame_index, "time_s": self.time_s,
            "mapped_revision": post["revision"], "metrics_before": deepcopy(state["metrics"]),
            "metrics_after": deepcopy(post["metrics"]),
            "all_users_effective": post["metrics"]["effective_users"] == post["metrics"]["admitted_users"],
            "irrecoverable_user_ids": deepcopy(plan["irrecoverable_user_ids"]),
        }
        return deepcopy(self.result)

    def hold(self, reason):
        state = self._fresh_context()
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A hold needs a nonempty evidence-based reason")
        self.result = {
            "decision_id": self.decision_id, "status": "held", "scenario": self.scene,
            "frame_index": self.frame_index, "time_s": self.time_s,
            "based_on_revision": state["revision"], "mapped_revision": state["revision"],
            "trigger_reason": self.trigger_reason, "reason": reason.strip(), "changes": [],
            "metrics_before": deepcopy(state["metrics"]), "metrics_after": deepcopy(state["metrics"]),
        }
        return deepcopy(self.result)

    def call(self, name, args, origin="model"):
        dispatch = {
            "get_decision_context": self.get_decision_context, "get_history": self.get_history,
            "get_bandwidth_options": self.get_bandwidth_options,
            "plan_reallocation": self.plan_reallocation, "commit_reallocation": self.commit_reallocation,
            "hold": self.hold,
        }
        started = time.perf_counter()
        full_evidence = False
        try:
            if name not in dispatch:
                raise ValueError("Unknown dynamic tool: " + str(name))
            if not isinstance(args, dict):
                raise ValueError("Tool args must be an object")
            forwarded = dict(args)
            full_evidence = forwarded.pop("full_evidence", False)
            if not isinstance(full_evidence, bool):
                raise ValueError("full_evidence must be boolean")
            result = dispatch[name](**forwarded)
        except (ValueError, TypeError, KeyError) as exc:
            result = {"error": str(exc)}
        self.tool_calls += 1
        self.model_tool_calls += int(origin == "model")
        view = deepcopy(result) if full_evidence else _model_view(name, result)
        _write_jsonl(self.output / "tool_calls.jsonl", {
            "decision_id": self.decision_id, "scenario": self.scene,
            "frame_index": self.frame_index, "time_s": self.time_s,
            "tool": name, "args": args, "origin": origin, "result": result,
            "model_observation": view, "tool_version": self.version,
            "tool_wall_seconds": time.perf_counter() - started,
        })
        return result, _packed(view)


class DynamicNetworkRAGAgent(NetworkRAGAgent):
    """Reuse the original loop/parser while adapting its task and tool vocabulary."""

    def __init__(self, llm, tools, max_iterations=8):
        self.llm, self.tools, self.max_iterations = llm, tools, max_iterations
        self.memory_handler = None
        self.chat_history = []
        self.graph_schema = _packed({
            "backend": "SQLite temporal property graph",
            "relationships": {name: list(pair) for name, pair in RELATION_TYPES.items()
                              if name not in ("INSTANCE_OF", "SCHEMA_RELATION")},
        })

    def _decide_next_action(self, question, scratchpad):
        prompt = f"""You are the TKG-DTN dynamic wireless service-assurance agent.
Follow Thought -> Action -> Observation. This is a fixed cohort of existing
subscribers with known service contracts, monitored at 10-second simulation frames.
Their current CQI may change with movement. The graph context is already prefetched.
Read the supplied present/past graph evidence and use the tools to decide an action.

Priorities: maximize the number of effective users in the fixed admitted cohort,
then maximize effective throughput. Fairness is reported, not the optimization goal.
Preserve every admitted user, each service category, all contracts and pool capacities.
No admissions, evictions, category switching, invented measurements or future data.
The exact plan_reallocation tool handles integer resources. Prefer previewing a plan
instead of guessing bandwidth. When a valid changed plan improves these priorities,
commit its observed plan_id. A no-change plan should be followed by hold(reason).
When full QoS is impossible, inspect the certificate: a valid best-effort plan can
improve other users while explicitly retaining the irrecoverable user's breach.
Never report successful recovery unless returned by a successful commit tool.
You may hold based on observed evidence; do not invent a plan ID or tool result.
commit_reallocation and hold are terminal actions. There is no separate finish tool.
The existing simplified CQI-to-rate model is used; latency is unmeasured.
This run uses the actual local SQLite graph; no Neo4j/Cypher execution is available.

Tools:
{TOOL_HELP}

Graph schema:
{self.graph_schema}

Question: {question}
Observations from this decision:
{scratchpad or '(none yet)'}

Return exactly one next action as:
思考: <brief decision rationale>
行动: {{"tool": "tool_name", "args": {{}}}}
"""
        return self.llm.complete([
            {"role": "system", "content": "Use only current/past observed graph facts and listed tools. Preserve the fixed service contracts."},
            {"role": "user", "content": prompt},
        ])


class DynamicTKGPolicy:
    """Run one explicit model decision against the already synchronized frame."""

    def __init__(self, graph, llm, output, max_iterations=8):
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.graph, self.llm, self.output = graph, llm, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.tools = DynamicWirelessTools(graph, self.output)
        self.agent = DynamicNetworkRAGAgent(llm, self.tools, max_iterations=max_iterations)

    def step(self, scene, trigger_reason):
        self.tools.start(scene, trigger_reason)
        self.llm.user_id = f"{scene}:frame_{self.tools.frame_index:03d}"
        started = time.perf_counter()
        call_before = getattr(self.llm, "calls", 0)
        question = (
            f"Scene {scene}, frame {self.tools.frame_index}, time {self.tools.time_s} s: "
            f"{trigger_reason}. Decide whether to reallocate resources or hold using the current graph."
        )
        try:
            for message in self.agent.ask_stream(question):
                event = json.loads(message.removeprefix("data: ").strip())
                _write_jsonl(self.output / "agent_events.jsonl", {
                    "decision_id": self.tools.decision_id, "scenario": scene,
                    "frame_index": self.tools.frame_index, "time_s": self.tools.time_s, **event,
                })
            if self.tools.result is None:
                raise RuntimeError("Dynamic ReAct ended without a terminal model decision")
        except Exception as exc:
            # The transport already sanitizes network errors. Still redact if a
            # downstream exception contains a credential, without logging settings.
            error = str(exc)
            settings = getattr(self.llm, "settings", {})
            if isinstance(settings, dict) and settings.get("api_key"):
                error = error.replace(str(settings["api_key"]), "[REDACTED]")
            _write_jsonl(self.output / "decisions.jsonl", {
                "decision_id": self.tools.decision_id, "scenario": scene,
                "frame_index": self.tools.frame_index, "time_s": self.tools.time_s,
                "status": "failed", "error": error,
                "decision_wall_seconds": time.perf_counter() - started,
                "llm_calls": getattr(self.llm, "calls", 0) - call_before,
                "tool_calls": self.tools.tool_calls, "model_tool_calls": self.tools.model_tool_calls,
            })
            raise
        result = deepcopy(self.tools.result) | {
            "decision_wall_seconds": time.perf_counter() - started,
            "llm_calls": getattr(self.llm, "calls", 0) - call_before,
            "tool_calls": self.tools.tool_calls, "model_tool_calls": self.tools.model_tool_calls,
        }
        _write_jsonl(self.output / "decisions.jsonl", result)
        return result
