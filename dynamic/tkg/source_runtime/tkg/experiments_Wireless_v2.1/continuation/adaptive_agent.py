"""On-demand evidence tools for the unchanged history-aware wireless optimizer.

The initial observation reads only the current graph state. Detailed user,
event, traffic and candidate evidence is fetched when the model requests it.
Resource optimization and transactional validation remain in the original
fairness implementation. A quality guard rejects unexamined holds and omitted
strict improvements, but never plans or commits on the model's behalf.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

from .agent import DynamicNetworkRAGAgent, DynamicTKGPolicy, _packed, _write_jsonl
from .fair_agent import FairWirelessTools, _fair_model_view
from .fairness import OBJECTIVE_PRIORITY, compare_scores
from .memory_view import record_table
from .service import SLICE_LIMITS


ADAPTIVE_VIEW_VERSION = "adaptive_evidence_on_demand_v2"
MAX_BATCH_QUERIES = 8
ADAPTIVE_TOOL_HELP = """
get_decision_context(detail='summary'|'full'): refresh current revision; summary is prefetched.
get_full_context(): fetch current state, recent events and traffic totals in compact tables.
inspect_users(user_ids?: list, include_history?: bool, window_s?: number): selected
  current users/contracts/allocations; optional completed-history totals. Omit IDs for all users.
get_constraints(): fixed resource rules, pool limits/usage and objective configuration.
get_history(user_id?, event_type?, through_time_s?, limit?): observed events only; limit 1..100.
get_traffic_history(user_id?, window_s?): completed traffic; single-user interval samples.
get_fairness_status(window_s?): historical cohort fairness; query windows do not change the objective.
get_bandwidth_options(user_id): exact current individual bandwidth choices.
query_network(queries): batch 1..8 read-only {tool,args} queries; no nested batches or actions.
plan_fair_reallocation(): exact joint optimizer; returns a small before/after certificate and plan_id.
inspect_plan(plan_id, user_ids?): compact candidate changes, projected users and certificates.
commit_fair_reallocation(plan_id): explicitly execute a fresh, valid, changed, strictly improving plan.
hold(reason): terminal only after a fresh exact preview shows no valid changed strict improvement.
Tool errors are observations: correct the request. No tool silently plans or commits.
Default observations are compact. Tables use *_columns/*_rows; *_shared applies to
every row and extra holds optional fields. No selected user or event is sampled away.
full_evidence=true returns the complete raw result, which can be large, especially
full-context traffic intervals. Request raw evidence only when that detail is needed.
"""


def _small_metrics(metrics):
    keys = ("observed_users", "admitted_users", "effective_users", "degraded_users",
            "unadmitted_users", "raw_throughput_mbps", "effective_throughput_mbps",
            "slice_usage_mhz", "latency_assessed")
    return {key: deepcopy(metrics[key]) for key in keys if key in metrics}


def _plan_certificate(plan):
    keys = ("plan_id", "scenario", "frame_index", "time_s", "revision",
            "history_through_time_s", "history_through_revision", "history_window_s",
            "planning_horizon_s", "retained_history_duration_s",
            "retained_history_window_start_s", "resource_valid", "changed",
            "objective_improves", "all_users_effective", "violations",
            "irrecoverable_user_ids", "inherited_wrong_slice_user_ids",
            "degraded_user_ids", "projection_semantics")
    value = {key: deepcopy(plan[key]) for key in keys if key in plan}
    value.update({
        "objective_before": deepcopy(plan["score_before"]["objective"]),
        "objective_after": deepcopy(plan["score"]["objective"]),
        "metrics_before": _small_metrics(plan["metrics_before"]),
        "metrics_after": _small_metrics(plan["score"]["metrics"]),
        "change_count": len(plan["changes"]),
        "total_bandwidth_change_mhz": plan["score"]["total_bandwidth_change_mhz"],
    })
    return value


def _adaptive_model_view(name, result):
    """Summarize only documented default views; preserve raw logged evidence."""
    if "error" in result:
        return deepcopy(result)
    if name == "plan_fair_reallocation":
        return _plan_certificate(result)
    if name == "get_full_context" or (name == "get_decision_context" and result.get("detail") == "full"):
        return _fair_model_view("get_decision_context", result)
    if name == "inspect_users":
        value = deepcopy(result)
        users = value.pop("users")
        rows = []
        for uid, record in users.items():
            current, allocation, contract = record["current"], record["allocation"], record["contract"]
            service = current["service"]
            rows.append({
                "user_id": uid, "admitted": record["admitted"],
                "assigned_slice": allocation["slice_type"] if allocation is not None else None,
                "required_slice": contract["slice_type"], "cqi": current["cqi"],
                "bandwidth_mhz": service["bandwidth_mhz"], "rate_mbps": service["rate_mbps"],
                "min_rate_mbps": contract["min_rate_mbps"], "max_rate_mbps": contract["max_rate_mbps"],
                "effective": service["effective"], "violations": service["violations"],
            })
        value.update(record_table(rows, "user"))
        if result["history_included"]:
            value.update(record_table([record["history"] for record in users.values()
                                       if record["history"] is not None], "history_user"))
            value["history_unavailable_user_ids"] = [uid for uid, record in users.items()
                                                       if record["history"] is None]
        return value
    if name == "inspect_plan":
        value = _fair_model_view("plan_fair_reallocation", result)
        # A projection inspection must retain the requested per-user evidence,
        # including both baseline and proposed values, at full numeric precision.
        for score_key in ("score_before", "score"):
            projections = result[score_key]["projected_per_user"]
            value[score_key].update(record_table(list(projections.values()), "projected_user"))
        return value
    if name == "query_network":
        value = deepcopy(result)
        for query in value["queries"]:
            if not isinstance(query["args"], dict) or not query["args"].get("full_evidence", False):
                query["result"] = _adaptive_model_view(query["tool"], query["result"])
        return value
    if name in ("get_history", "get_traffic_history", "get_fairness_status",
                "get_bandwidth_options"):
        return _fair_model_view(name, result)
    return deepcopy(result)


class AdaptiveFairWirelessTools(FairWirelessTools):
    version = "dynamic_history_proportional_fair_adaptive_v1"

    READ_ONLY_TOOLS = frozenset({
        "get_decision_context", "get_full_context", "inspect_users", "get_constraints",
        "get_history", "get_traffic_history", "get_fairness_status",
        "get_bandwidth_options", "inspect_plan",
    })

    def _dispatch(self):
        return {name: getattr(self, name) for name in (
            *sorted(self.READ_ONLY_TOOLS), "query_network", "plan_fair_reallocation",
            "commit_fair_reallocation", "hold",
        )}

    def _same_revision(self, state):
        current = self._current()
        if current["revision"] != state["revision"]:
            self.context_read, self.context_revision = False, None
            raise ValueError("Graph revision changed during query; refresh get_decision_context")

    @staticmethod
    def _user_ids(user_ids, known):
        if user_ids is None:
            return list(known)
        if not isinstance(user_ids, list) or not user_ids:
            raise ValueError("user_ids must be a nonempty list or omitted for all users")
        if any(isinstance(uid, bool) or not isinstance(uid, (str, int)) for uid in user_ids):
            raise ValueError("Each user ID must be a string or integer")
        ids = [str(uid) for uid in user_ids]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate user IDs are not allowed")
        if any(uid not in known for uid in ids):
            raise ValueError("Unknown user ID in this query")
        return ids

    def get_decision_context(self, detail="summary"):
        if detail not in ("summary", "full"):
            raise ValueError("detail must be 'summary' or 'full'")
        if detail == "full":
            return self.get_full_context()
        # Do not call the parent context reader: it materializes every traffic
        # interval and event before compacting them. This path is genuinely lazy.
        state = self._current()
        admitted = state["allocations"]
        result = {
            "detail": "summary", "decision_id": self.decision_id,
            "run_id": self.graph.run_id, "backend": "sqlite_local_property_graph",
            "trigger_reason": self.trigger_reason,
            "state": {
                **{key: state[key] for key in ("scenario", "frame_index", "time_s", "revision")},
                "phase": state.get("phase", "post_control"),
                "metrics": _small_metrics(state["metrics"]),
                "observed_user_count": len(state["users"]),
                "admitted_user_count": len(admitted),
                "degraded_user_ids": [uid for uid in admitted
                                      if not state["users"][uid]["service"]["effective"]],
                "inherited_wrong_slice_user_ids": [uid for uid, allocation in admitted.items()
                    if allocation["slice_type"] != state["contracts"][uid]["slice_type"]],
                "inherited_unadmitted_user_ids": [uid for uid in state["users"] if uid not in admitted],
                "active_episode_user_ids": list(state["active_episodes"]),
            },
            "history_boundary": {"through_time_s": state["time_s"],
                                 "through_revision": state["revision"]},
            "history_detail_loaded": False,
            "fairness_history_window_s": self.history_window_s,
            "planning_horizon_s": self.horizon_s,
            "objective_priority": list(OBJECTIVE_PRIORITY),
            "evidence_semantics": "Only completed intervals are historical traffic. No historical values are inferred by this summary; request evidence as needed. Plans project current CQI, not future measurements.",
            "decision_guard": "An exact preview is required before hold; a valid changed strict improvement must be explicitly committed. No automatic execution.",
        }
        self._same_revision(state)
        self.context_read, self.context_revision = True, state["revision"]
        return result

    def get_full_context(self):
        result = super().get_decision_context()
        result["detail"] = "full"
        return result

    def inspect_users(self, user_ids=None, include_history=False, window_s=None):
        state = self._fresh_context()
        if not isinstance(include_history, bool):
            raise ValueError("include_history must be boolean")
        if window_s is not None and not include_history:
            raise ValueError("window_s requires include_history=true")
        ids = self._user_ids(user_ids, state["users"])
        users = {uid: {
            "user_id": uid, "admitted": uid in state["allocations"],
            "current": deepcopy(state["users"][uid]),
            "allocation": deepcopy(state["allocations"].get(uid)),
            "contract": deepcopy(state["contracts"][uid]),
        } for uid in ids}
        result = {"scenario": self.scene, "time_s": state["time_s"],
                  "revision": state["revision"], "users": users,
                  "history_included": include_history}
        if include_history:
            history = self._traffic(state, user_id=ids[0] if len(ids) == 1 else None,
                                    window_s=window_s)
            result["history_boundary"] = {key: deepcopy(history[key]) for key in (
                "through_time_s", "through_revision", "window_start_s", "window_s", "duration_s")}
            result["optimization_history_window_s"] = self.history_window_s
            result["query_changes_optimization_window"] = False
            for uid in ids:
                users[uid]["history"] = deepcopy(history["users"].get(uid))
            result["history_semantics"] = "Completed simulated traffic totals; null history denotes an unadmitted user. Interval records are available from get_traffic_history."
        self._same_revision(state)
        return result

    def get_constraints(self):
        state = self._fresh_context()
        return {
            "scenario": self.scene, "time_s": state["time_s"], "revision": state["revision"],
            "constraints": deepcopy(SLICE_LIMITS),
            "current_pool_usage_mhz": deepcopy(state["metrics"]["slice_usage_mhz"]),
            "objective_priority": list(OBJECTIVE_PRIORITY),
            "history_window_s": self.history_window_s, "planning_horizon_s": self.horizon_s,
            "immutable": ["admitted cohort", "assigned slices", "true service contracts", "current CQI"],
            "bandwidth_control": "Integer MHz within assigned-slice bounds, contract rate ceilings and joint pool capacities.",
            "wrong_slice_semantics": "Bandwidth can change; assigned/required slices remain distinct. Wrong-slice users enter raw-traffic utility but never count as effective.",
            "latency_assessed": False, "jain_is_evaluation_only": True,
            "max_batch_queries": MAX_BATCH_QUERIES,
        }

    def _fresh_plan(self, plan_id, state):
        if not isinstance(plan_id, str) or plan_id not in self.plans:
            raise ValueError("Unknown plan_id; request plan_fair_reallocation first")
        plan = self.plans[plan_id]
        if (plan["scenario"], plan["frame_index"], plan["time_s"], plan["revision"]) != (
                self.scene, state["frame_index"], state["time_s"], state["revision"]):
            raise ValueError("Stale plan revision; preview again from current context")
        return plan

    def inspect_plan(self, plan_id, user_ids=None):
        state = self._fresh_context()
        plan = self._fresh_plan(plan_id, state)
        ids = self._user_ids(user_ids, state["allocations"])
        result = {key: deepcopy(value) for key, value in plan.items() if key != "allocations"}
        if user_ids is not None:
            selected = set(ids)
            result["total_change_count"] = len(result["changes"])
            result["changes"] = [row for row in result["changes"] if str(row["user_id"]) in selected]
            for score in (result, result["score"], result["score_before"]):
                if "projected_per_user" in score:
                    score["projected_per_user"] = {uid: row for uid, row in score["projected_per_user"].items() if uid in selected}
                for metric in ("historical_fairness", "projected_fairness"):
                    if metric in score and "users" in score[metric]:
                        score[metric]["users"] = {uid: row for uid, row in score[metric]["users"].items() if uid in selected}
            result["detail_scope"] = {"user_ids": ids, "aggregate_metrics_scope": "entire fixed admitted cohort"}
        self._same_revision(state)
        return result

    def query_network(self, queries):
        state = self._fresh_context()
        if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_BATCH_QUERIES:
            raise ValueError(f"queries must contain 1..{MAX_BATCH_QUERIES} read-only requests")
        # Validate the complete action allowlist before running any item.
        for query in queries:
            if not isinstance(query, dict) or set(query) - {"tool", "args"}:
                raise ValueError("Each query must be an object containing tool and optional args")
            if not isinstance(query.get("tool"), str) or query["tool"] not in self.READ_ONLY_TOOLS:
                raise ValueError("Batch accepts read-only tools only; planning, terminal actions and nested batches are forbidden")
        dispatch, results = self._dispatch(), []
        for query in queries:
            args = deepcopy(query.get("args", {}))
            try:
                if not isinstance(args, dict):
                    raise ValueError("Query args must be an object")
                forwarded = dict(args)
                full_evidence = forwarded.pop("full_evidence", False)
                if not isinstance(full_evidence, bool):
                    raise ValueError("full_evidence must be boolean")
                result = dispatch[query["tool"]](**forwarded)
            except (ValueError, TypeError, KeyError) as exc:
                result = {"error": str(exc)}
            results.append({"tool": query["tool"], "args": args, "result": result})
        # Discard a mixed-revision observation instead of returning inconsistent evidence.
        self._same_revision(state)
        return {"scenario": self.scene, "time_s": state["time_s"],
                "revision": state["revision"], "queries": results}

    def commit_fair_reallocation(self, plan_id):
        state = self._fresh_context()
        plan = self._fresh_plan(plan_id, state)
        if not plan["resource_valid"]:
            raise ValueError("Hard-resource-invalid plans cannot be committed")
        if not plan["changed"]:
            raise ValueError("Plan makes no allocation changes; use hold with a reason")
        if compare_scores(plan["score"], plan["score_before"]) <= 0:
            raise ValueError("Plan must strictly improve the ordered objectives before commit")
        return super().commit_fair_reallocation(plan_id)

    def hold(self, reason):
        state = self._fresh_context()
        fresh = [plan for plan in self.plans.values()
                 if (plan["scenario"], plan["frame_index"], plan["time_s"], plan["revision"]) ==
                 (self.scene, state["frame_index"], state["time_s"], state["revision"])]
        if not fresh:
            raise ValueError("Hold requires a fresh exact preview; call plan_fair_reallocation first")
        if any(plan["resource_valid"] and plan["changed"]
               and compare_scores(plan["score"], plan["score_before"]) > 0 for plan in fresh):
            raise ValueError("A fresh valid changed plan strictly improves the ordered objectives; explicitly commit its plan_id instead of hold")
        return super().hold(reason)

    def call(self, name, args, origin="model"):
        started, full_evidence = time.perf_counter(), False
        try:
            dispatch = self._dispatch()
            if not isinstance(name, str) or name not in dispatch:
                raise ValueError("Unknown adaptive fairness tool: " + str(name))
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
        view = deepcopy(result) if full_evidence else _adaptive_model_view(name, result)
        _write_jsonl(self.output / "tool_calls.jsonl", {
            "decision_id": self.decision_id, "scenario": self.scene,
            "frame_index": self.frame_index, "time_s": self.time_s,
            "tool": name, "args": args, "origin": origin, "result": result,
            "model_observation": view, "tool_version": self.version,
            "model_view_version": "full_evidence" if full_evidence else ADAPTIVE_VIEW_VERSION,
            "tool_wall_seconds": time.perf_counter() - started,
        })
        return result, _packed(view)


class AdaptiveFairNetworkRAGAgent(DynamicNetworkRAGAgent):
    """Keep the existing parser, streaming ReAct loop and terminal execution truth."""

    def _decide_next_action(self, question, scratchpad):
        if getattr(self, "_protocol_decision_id", None) != self.tools.decision_id:
            self._protocol_decision_id = self.tools.decision_id
            self._protocol_repair_used = False

        prompt = f"""Control the current TKG-DTN wireless frame using graph evidence and tools.
Only a small current-state summary is prefetched. Choose any needed evidence
queries, individually or in one read-only batch; no fixed retrieval sequence is required.
The optimizer itself reads the complete current state and configured history, so
its inputs need not be queried first. Additional reads can resolve specific evidence questions.

Keep the fixed admitted cohort, assigned slices and true contracts. Inherited
unadmitted/wrong-slice users remain so; wrong-slice bandwidth can change but cannot
repair classification. Rates are simulated, latency unmeasured. Use only past/current
evidence: completed traffic is history; constant-current-CQI next-interval projections
are hypothetical, and same-time commits never create past traffic.

The unchanged exact optimizer orders: effective-user count, then sum-log projected
historical mean throughput, then effective throughput, then minimum bandwidth change.
Jain is an evaluation metric. The configured history window/horizon is fixed.
Before a terminal choice obtain an exact preview. Explicitly commit a fresh valid
changed strict improvement; hold when no such improvement exists, with an evidenced
reason. Both end the decision. Guard errors require a corrected model action;
there is no automatic planning, commit or fallback. Never invent plan IDs or results.

Tools:
{ADAPTIVE_TOOL_HELP}

Question: {question}
Observations:
{scratchpad or '(none yet)'}

Reply only in ordinary text using the two lines below and exactly one JSON action.
Do not use XML, DSML, native tool-call markup, code fences or multiple actions.
To request several reads, use one query_network action whose args contain queries.
思考: <brief evidence-based rationale>
行动: {{"tool": "tool_name", "args": {{}}}}
"""
        messages = [
            {"role": "system", "content": "Choose evidence on demand. Preserve contracts and ordered objectives. Projections are not measurements. Only explicit tools execute decisions. Reply as ordinary text: 思考: rationale then 行动: one JSON object with a string tool and object args. Never emit XML, DSML, native tool-call markup or multiple actions."},
            {"role": "user", "content": prompt},
        ]

        def validate(output):
            if not isinstance(output, str):
                raise ValueError("Model output must be ordinary text")
            _, action = self._parse_llm_action(output)
            if not isinstance(action, dict):
                raise ValueError("Action must be one JSON object")
            if not isinstance(action.get("tool"), str) or not action["tool"].strip():
                raise ValueError("Action tool must be a nonempty string")
            if action["tool"].strip() == "finish":
                raise ValueError("finish is unavailable; only commit_fair_reallocation or hold can end a decision through the terminal guards")
            if not isinstance(action.get("args"), dict):
                raise ValueError("Action args must be a JSON object")

        def log(event, output, error, *, is_retry, retry_scheduled=False):
            _write_jsonl(self.tools.output / "protocol_events.jsonl", {
                "decision_id": self.tools.decision_id, "scenario": self.tools.scene,
                "frame_index": self.tools.frame_index, "time_s": self.tools.time_s,
                "origin": "protocol_repair", "event": event,
                "raw_output": output, "error": error, "is_retry": is_retry,
                "retry_scheduled": retry_scheduled,
            })

        output = self.llm.complete(messages)
        try:
            validate(output)
        except (ValueError, TypeError) as exc:
            error = str(exc)
            can_retry = not self._protocol_repair_used
            log("parse_error", output, error, is_retry=False, retry_scheduled=can_retry)
            if not can_retry:
                raise ValueError("Action format invalid; this decision's single protocol repair was already used: " + error) from exc
            # This is an actual model call, not an inferred action or hidden tool
            # execution. The existing transport accounts for its latency/tokens.
            self._protocol_repair_used = True
            messages.extend([
                {"role": "assistant", "content": output if isinstance(output, str) else str(output)},
                {"role": "user", "content": (
                    "Your response was not a valid action and no tool from it was executed. "
                    "Format error: " + error + ". Return exactly one next action in ordinary text:\n"
                    "思考: <brief rationale>\n"
                    '行动: {"tool": "tool_name", "args": {}}\n'
                    "The tool must be a nonempty string and args a JSON object. "
                    "No XML, DSML, native tool-call markup, code fences or multiple actions. "
                    "For several reads choose one query_network JSON action. "
                    "Use the same observed evidence; do not invent tool results."
                )},
            ])
            log("repair_request", output, error, is_retry=True, retry_scheduled=True)
            repaired = self.llm.complete(messages)
            try:
                validate(repaired)
            except (ValueError, TypeError) as repair_exc:
                log("repair_failed", repaired, str(repair_exc), is_retry=True)
                raise ValueError("Action format invalid after one protocol repair: " + str(repair_exc)) from repair_exc
            log("repair_succeeded", repaired, None, is_retry=True)
            return repaired
        return output


class AdaptiveFairDynamicTKGPolicy(DynamicTKGPolicy):
    def __init__(self, graph, llm, output, max_iterations=8, *, horizon_s=None, history_window_s=None):
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.graph, self.llm, self.output = graph, llm, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.tools = AdaptiveFairWirelessTools(graph, self.output, horizon_s=horizon_s,
                                               history_window_s=history_window_s)
        self.agent = AdaptiveFairNetworkRAGAgent(llm, self.tools, max_iterations=max_iterations)
