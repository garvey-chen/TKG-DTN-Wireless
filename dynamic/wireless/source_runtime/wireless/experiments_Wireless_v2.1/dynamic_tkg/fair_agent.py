"""History-aware service assurance using the existing TKG-DTN ReAct loop.

Only graph-finalized intervals are evidence. The mathematical resource tool
projects one next interval under current CQI; projections are never traffic
measurements and are never inserted into the historical traffic ledger.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time
import uuid

from .agent import (
    DynamicNetworkRAGAgent, DynamicTKGPolicy, DynamicWirelessTools,
    _model_view, _packed, _write_jsonl,
)
from .fairness import compare_scores, optimize_fair_service, score_candidate
from .fairness_schema import RELATION_TYPES


FAIR_TOOL_HELP = """
get_decision_context(): Refresh current graph state, service contracts, recent
  events, and graph-backed traffic/fairness for already completed intervals.
  These facts are prefetched at the start of this decision.
get_history(user_id?: str, event_type?: str, through_time_s?: number, limit?: int):
  Read observed past/current graph events. No future time is allowed.
get_traffic_history(user_id?: str, window_s?: number): Read completed traffic
  intervals and cumulative per-user simulated traffic, SLA deficit and outage.
  Default window is the experiment's configured fairness window; a positive
  window_s only changes this query, never the optimization objective.
  A single-user query includes up to 30 most recent completed interval samples;
  full_evidence=true includes all matching interval records.
get_fairness_status(window_s?: number): Read historical average-throughput Jain
  indices and per-user history for the fixed admitted cohort. At time zero
  historical averages/fairness are undefined, not fabricated observations.
get_bandwidth_options(user_id: str): List current integer-bandwidth QoS choices.
plan_fair_reallocation(): Nonmutating exact integer-resource preview. Priorities:
  (1) maximize current effective users; (2) maximize sum-log projected historical
  average throughput; (3) maximize effective throughput; (4) minimize bandwidth
  change. Uses graph-backed completed traffic and the configured fixed window.
  The next interval's CQI is assumed equal to current CQI for this projection.
  Returns a fresh plan_id, revision, proposed changes, scores and certificates.
commit_fair_reallocation(plan_id: str): Explicitly commit a fresh changed valid
  plan. Ends this decision. Historical traffic is not changed by a same-time
  allocation commit; it accrues only when the next interval closes.
hold(reason: str): End this decision without changing resources. Give a specific
  reason supported by current graph evidence or the preview certificate.
Every tool accepts full_evidence=true to return complete logged evidence.
"""


def _history_view(history, *, interval_samples=False):
    """Keep all actual per-user service evidence without repeated JSON keys."""
    value = deepcopy(history)
    users = value.pop("users", None)
    if users is not None:
        columns = ["user_id", "slice_type", "served_mbit", "effective_served_mbit",
                   "observed_duration_s", "mean_rate_mbps", "normalized_mean_rate",
                   "deficit_mbit", "effective_time_s", "min_rate_mbps"]
        value["user_columns"] = columns
        value["user_rows"] = [[uid] + [row.get(key) for key in columns[1:]]
                              for uid, row in users.items()]
    # Full interval records remain in the tool log and on-demand full evidence.
    if isinstance(value.get("intervals"), list):
        intervals = value.pop("intervals")
        value["returned_interval_count"] = len(intervals)
        if interval_samples and users is not None and len(users) == 1:
            columns = ["overlap_start_time_s", "overlap_end_time_s", "modeled_rate_mbps",
                       "included_traffic_mbit", "service_effective", "source_revision", "recorded_revision"]
            value["interval_columns"] = columns
            value["interval_rows"] = [[row[key] for key in columns] for row in intervals[-30:]]
            value["interval_samples_truncated"] = len(intervals) > 30
    return value


def _fair_model_view(name, result):
    if "error" in result:
        return deepcopy(result)
    if name == "get_decision_context":
        value = _model_view(name, result)
        value["traffic_history"] = _history_view(value["traffic_history"])
        return value
    if name in ("get_traffic_history", "get_fairness_status"):
        return _history_view(result, interval_samples=name == "get_traffic_history")
    value = deepcopy(result)
    if name == "plan_fair_reallocation":
        # Per-user projections are available in the full log. Preserve the
        # explicit historical-versus-hypothetical aggregate distinction here.
        for key in ("score_before", "score"):
            if key in value:
                value[key].pop("projected_per_user", None)
                for metric in ("historical_fairness", "projected_fairness"):
                    value[key].get(metric, {}).pop("users", None)
        for metric in ("historical_fairness", "projected_fairness"):
            value.get(metric, {}).pop("users", None)
        if isinstance(value.get("evidence"), dict):
            value["evidence"].pop("per_slice", None)
    elif name == "get_bandwidth_options":
        value.pop("rates_mbps", None)
        value["scope"] = "Individual QoS choices; plan_fair_reallocation checks joint pool feasibility."
    elif name == "get_history":
        value = _model_view(name, result)
    return value


class FairWirelessTools(DynamicWirelessTools):
    version = "dynamic_history_proportional_fair_exact_v1"

    def __init__(self, graph, output, *, horizon_s=None, history_window_s=None):
        super().__init__(graph, output)
        self.horizon_s = getattr(graph, "horizon_s", 10.0) if horizon_s is None else horizon_s
        self.history_window_s = (getattr(graph, "history_window_s", None)
                                 if history_window_s is None else history_window_s)
        for name, value in (("horizon_s", self.horizon_s),
                            ("history_window_s", self.history_window_s)):
            if value is None and name == "history_window_s":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.history_window_s != getattr(graph, "history_window_s", None):
            raise ValueError("Policy history window must match the graph's fixed experiment configuration")
        if self.horizon_s != getattr(graph, "horizon_s", 10.0):
            raise ValueError("Policy horizon must match the graph's fixed experiment configuration")
        self.context_revision = None

    def start(self, scene, trigger_reason):
        super().start(scene, trigger_reason)
        self.context_revision = None

    def _fresh_context(self):
        state = super()._fresh_context()
        if self.context_revision != state["revision"]:
            raise ValueError("Graph revision changed; refresh get_decision_context")
        return state

    def _traffic(self, state, *, user_id=None, window_s=None):
        return self.graph.traffic_history(
            self.scene, user_id=user_id,
            window_s=self.history_window_s if window_s is None else window_s,
            through_time_s=state["time_s"], through_revision=state["revision"])

    def get_decision_context(self):
        context = super().get_decision_context()
        state = context["state"]
        history = self._traffic(state)
        if self._current()["revision"] != state["revision"]:
            self.context_read = False
            raise ValueError("Graph changed while reading traffic evidence; refresh context")
        self.context_revision = state["revision"]
        context.update({
            "traffic_history": history,
            "fairness_history_window_s": self.history_window_s,
            "planning_horizon_s": self.horizon_s,
            "objective": ["maximize current effective users in the fixed admitted cohort",
                          "then maximize sum-log projected historical mean throughput",
                          "then maximize effective throughput", "then minimize bandwidth change"],
            "history_semantics": "Completed simulated traffic intervals only. A commit at this time adds no past traffic.",
            "projection_semantics": "Hypothetical next interval holds current CQI; projected traffic is never observed history.",
            "fairness_semantics": "Proportional-fair sum-log objective; Jain indices are evaluation metrics, not the optimized objective.",
        })
        return context

    def get_traffic_history(self, user_id=None, window_s=None):
        state = self._fresh_context()
        if user_id is not None and str(user_id) not in state["allocations"]:
            raise ValueError("Traffic fairness tools operate on the fixed admitted cohort only")
        return self._traffic(state, user_id=user_id, window_s=window_s)

    def get_fairness_status(self, window_s=None):
        state = self._fresh_context()
        history = self._traffic(state, window_s=window_s)
        return {**history,
                "optimization_history_window_s": self.history_window_s,
                "query_changes_optimization_window": False,
                "current_effective_users": state["metrics"]["effective_users"],
                "fairness_objective": "sum-log projected historical average throughput after maximizing current effective users",
                "jain_is_evaluation_only": True}

    def plan_fair_reallocation(self):
        state = self._fresh_context()
        # Rolling windows expire their earliest data by the projected horizon.
        # This retains only history belonging to that hypothetical end-window.
        history = self.graph.planning_history(self.scene, horizon_s=self.horizon_s)
        if (history["through_time_s"], history["through_revision"]) != (state["time_s"], state["revision"]):
            raise ValueError("Graph changed while reading planning history; refresh context")
        cqis = {uid: user["cqi"] for uid, user in state["users"].items()}
        before_score = score_candidate(state["allocations"], cqis, state["contracts"],
                                       history, horizon_s=self.horizon_s,
                                       reference_allocations=state["allocations"])
        planned = optimize_fair_service(state["allocations"], cqis, state["contracts"],
                                        history, horizon_s=self.horizon_s)
        plan_id = uuid.uuid4().hex[:16]
        plan = {**deepcopy(planned), "plan_id": plan_id, "scenario": self.scene,
                "frame_index": state["frame_index"], "time_s": state["time_s"],
                "revision": state["revision"], "history_through_time_s": state["time_s"],
                "history_through_revision": state["revision"],
                "history_window_s": self.history_window_s, "planning_horizon_s": self.horizon_s,
                "retained_history_duration_s": history["duration_s"],
                "retained_history_window_start_s": history["window_start_s"],
                "metrics_before": deepcopy(state["metrics"]), "score_before": before_score,
                "changed": planned["allocations"] != state["allocations"],
                "objective_improves": compare_scores(planned["score"], before_score) > 0,
                "projection_semantics": "Proposed next-interval simulated service under constant current CQI, not observed traffic."}
        self.plans[plan_id] = plan
        return {key: deepcopy(value) for key, value in plan.items() if key != "allocations"}

    def commit_fair_reallocation(self, plan_id):
        state = self._fresh_context()
        if not isinstance(plan_id, str) or plan_id not in self.plans:
            raise ValueError("Unknown plan_id; request plan_fair_reallocation first")
        plan = self.plans[plan_id]
        if (plan["scenario"], plan["frame_index"], plan["revision"]) != (
                self.scene, state["frame_index"], state["revision"]):
            raise ValueError("Stale plan revision; preview again from current context")
        if not plan["resource_valid"]:
            raise ValueError("Hard-resource-invalid plans cannot be committed")
        if not plan["changed"]:
            raise ValueError("Plan makes no allocation changes; use hold with a reason")
        decision = {
            "decision_id": self.decision_id, "plan_id": plan_id,
            "agent": "TKG-DTN_dynamic_history_fairness", "trigger_reason": self.trigger_reason,
            "tool_version": self.version, "changes": deepcopy(plan["changes"]),
            "based_on_revision": state["revision"],
            "history_through_time_s": plan["history_through_time_s"],
            "history_through_revision": plan["history_through_revision"],
            "history_window_s": self.history_window_s, "planning_horizon_s": self.horizon_s,
            "objective_before": deepcopy(plan["score_before"]["objective"]),
            "objective_proposed": deepcopy(plan["score"]["objective"]),
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
            "projected_fairness": deepcopy(plan["score"]["projected_fairness"]),
            "projection_semantics": plan["projection_semantics"],
        }
        return deepcopy(self.result)

    def hold(self, reason):
        result = super().hold(reason)
        result.update({"agent": "TKG-DTN_dynamic_history_fairness", "tool_version": self.version,
                       "history_through_time_s": self.time_s,
                       "history_through_revision": result["based_on_revision"],
                       "history_window_s": self.history_window_s, "planning_horizon_s": self.horizon_s})
        self.result = result
        return deepcopy(result)

    def call(self, name, args, origin="model"):
        dispatch = {
            "get_decision_context": self.get_decision_context, "get_history": self.get_history,
            "get_bandwidth_options": self.get_bandwidth_options,
            "get_traffic_history": self.get_traffic_history, "get_fairness_status": self.get_fairness_status,
            "plan_fair_reallocation": self.plan_fair_reallocation,
            "commit_fair_reallocation": self.commit_fair_reallocation, "hold": self.hold,
        }
        started, full_evidence = time.perf_counter(), False
        try:
            if name not in dispatch:
                raise ValueError("Unknown dynamic fairness tool: " + str(name))
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
        view = deepcopy(result) if full_evidence else _fair_model_view(name, result)
        _write_jsonl(self.output / "tool_calls.jsonl", {
            "decision_id": self.decision_id, "scenario": self.scene,
            "frame_index": self.frame_index, "time_s": self.time_s,
            "tool": name, "args": args, "origin": origin, "result": result,
            "model_observation": view, "tool_version": self.version,
            "tool_wall_seconds": time.perf_counter() - started,
        })
        return result, _packed(view)


class FairNetworkRAGAgent(DynamicNetworkRAGAgent):
    """Retain the original parser, stream loop and execution-truth final answer."""

    def __init__(self, llm, tools, max_iterations=8):
        super().__init__(llm, tools, max_iterations=max_iterations)
        self.graph_schema = _packed({
            "backend": "SQLite temporal property graph with immutable modeled TrafficInterval nodes",
            "relationships": {name: list(pair) for name, pair in RELATION_TYPES.items()
                              if name not in ("INSTANCE_OF", "SCHEMA_RELATION")},
        })

    def _decide_next_action(self, question, scratchpad):
        prompt = f"""You are the TKG-DTN dynamic wireless service-assurance agent.
Follow Thought -> Action -> Observation. The fixed admitted subscribers retain
their service categories and contracts. Graph facts and completed traffic history
are prefetched. Decisions occur every active 10-second interval, even if CQI did
not change: historical service balance can change while current channels do not.

Priorities are lexicographic: maximize current effective users FIRST; among those
plans maximize sum-log projected historical mean throughput (proportional fairness);
then maximize effective throughput and minimize resource changes. Never sacrifice
an effective user to improve fairness. Jain fairness is an evaluation index, not
the exact optimized utility. Dividing each user's rate by their fixed minimum
inside a logarithm does not create per-user priority weights.

Use graph-backed completed intervals as history. Never claim future bytes were
served. A plan projects one next interval using current CQI; its projection can
change next frame and is not an observed recovery guarantee. A same-time commit
changes current resources only; history remains unchanged until the interval ends.
The configured cumulative/window objective is fixed for this experiment; querying
a shorter history window does not change that objective. Past windows may reveal
which users have low mean throughput, service deficits or sustained outages.

Use plan_fair_reallocation for a joint resource preview. Commit its actual plan_id
when a valid changed plan improves the ordered objectives; hold for an unchanged
optimum or with another specific evidence-based reason. Both commit and hold end
the decision. Never invent plan IDs, outcomes, traffic or future CQI. No automatic
fallback exists. A best-effort plan may preserve an explicitly recorded breach
when current resources or the user's CQI cannot support their minimum rate.
Do not add/remove users, change slices/contracts or exceed pool/bandwidth limits.
Current rates and traffic are simulated under the existing CQI model; latency is
unmeasured. This is the actual local SQLite temporal graph; no Cypher execution.

Tools:
{FAIR_TOOL_HELP}

Graph schema:
{self.graph_schema}

Question: {question}
Observations from this decision:
{scratchpad or '(none yet)'}

Return exactly one next action as:
思考: <brief evidence-based rationale>
行动: {{"tool": "tool_name", "args": {{}}}}
"""
        return self.llm.complete([
            {"role": "system", "content": "Use current/past graph evidence. Maximize service count before history-aware fairness. Projections are not measurements."},
            {"role": "user", "content": prompt},
        ])


class FairDynamicTKGPolicy(DynamicTKGPolicy):
    def __init__(self, graph, llm, output, max_iterations=8, *, horizon_s=None, history_window_s=None):
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.graph, self.llm, self.output = graph, llm, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.tools = FairWirelessTools(graph, output, horizon_s=horizon_s,
                                       history_window_s=history_window_s)
        self.agent = FairNetworkRAGAgent(llm, self.tools, max_iterations=max_iterations)
