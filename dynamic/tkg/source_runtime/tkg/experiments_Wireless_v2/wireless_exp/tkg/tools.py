"""Graph tools with exact resource previews and version-checked execution (v2)."""

import json
import uuid
from copy import deepcopy

from ..metrics import SLICE_LIMITS, rate_mbps
from ..policies import load_legacy
from .executor import resource_violations
from .resources import bandwidth_options, describe_changes, optimize_fixed_users
from .model_view import model_view

TOOL_HELP = """
get_decision_context(): Refresh user, state, constraints and complete intent KB from Neo4j.
search_entities(keyword: str or list): Search arrived users, slices and schema types in this run.
execute_cypher(query: str): Scoped templates: 'current_state', 'schema', 'recent_events'.
Network_topology_query(): Read current resource allocations from Neo4j.
get_bandwidth_options(slice_type: 'eMBB'|'URLLC'): Enumerate all legal integer bandwidths and all
  feasible choices after reducing existing users no lower than QoS minima. Exact evidence.
plan_allocation(slice_type: 'eMBB'|'URLLC', bandwidth: optional integer): Exact nonmutating preview.
  Prefer OMITTING bandwidth: admits this user if feasible, retains all current users and slices,
  satisfies minima, and maximizes total rate for that fixed set. Explicit bandwidth pins only the
  new user. No nested model calls or evictions. Returns plan_id, revision, changes and feasibility.
commit_allocation(plan_id: str): Commit a valid, fresh admission plan and map it to Neo4j.
plan_reallocation(slice_type: optional 'eMBB'|'URLLC'): Preview optimal bandwidth for existing users
  only. Omitting slice_type optimizes both pools. Does not admit or remove anyone.
commit_reallocation(plan_id: str): Commit reallocation; current request still needs a decision.
get_history(revision: optional integer): Replay graph deltas at a revision, or list recent events.
reject_request(slice_type: 'eMBB'|'URLLC', reason: str): Only allowed when exact calculation proves
  no bandwidth can admit this user in that slice while retaining the current users.
finish(): Only after a committed admission or recorded rejection.
Any tool accepts full_evidence=true to return its uncompressed evidence.
"""


class WirelessTools:
    version = "exact_resources_v2"

    def __init__(self, graph, llm, output):
        self.graph, self.llm, self.output = graph, llm, output
        # Reuse only latency semantics, never the old LLM resource fallback.
        self.latency_heuristic = load_legacy("agent_kb").apply_heuristic_latency
        self.start(None)

    def start(self, user_id):
        self.user_id = str(user_id) if user_id is not None else None
        self.plans, self.result, self.context_read = {}, None, False

    def context(self):
        value = self.graph.context(self.user_id)
        self.context_read = True
        return value

    def _fresh_context(self):
        if not self.context_read:
            raise ValueError("Read get_decision_context before planning")
        if self.result is not None:
            raise ValueError("This request already has a terminal decision")
        return self.graph.context(self.user_id, reuse_verified=True)

    def get_bandwidth_options(self, slice_type):
        context = self._fresh_context()
        return bandwidth_options(context["state"], context["user"], slice_type) | {"revision": context["revision"]}

    def _save_plan(self, kind, context, optimized, slice_type=None, feasibility=None):
        plan_id = uuid.uuid4().hex[:12]
        plan = {"plan_id": plan_id, "kind": kind, "revision": context["revision"],
                "slice_type": slice_type, "resource_engine": self.version,
                "objective": "retain fixed users and slices; satisfy QoS; maximize total rate", **optimized}
        if feasibility is not None:
            plan["feasibility_certificate"] = feasibility
        if optimized["feasible"]:
            candidate = optimized["candidate_state"]
            plan.update(describe_changes(context["state"], candidate))
            if kind == "admission":
                plan["allocation"] = next(deepcopy(a) for a in candidate[SLICE_LIMITS[slice_type]["key"]]["users"]
                                          if str(a["user_id"]) == self.user_id)
        self.plans[plan_id] = plan
        # Keep full executable state private; return compact evidence to the model.
        return {k: deepcopy(v) for k, v in plan.items() if k != "candidate_state"}

    def plan_allocation(self, slice_type, bandwidth=None):
        context = self._fresh_context()
        certificate = bandwidth_options(context["state"], context["user"], slice_type)
        if bandwidth is not None and (isinstance(bandwidth, bool) or not isinstance(bandwidth, int)):
            raise ValueError("Bandwidth must be an integer number of MHz")
        if not certificate["admission_feasible"]:
            return self._save_plan("admission", context, {"feasible": False, "violations": ["no_feasible_integer_admission"]}, slice_type, certificate)
        candidate = deepcopy(context["state"])
        user, limits = context["user"], SLICE_LIMITS[slice_type]
        b = certificate["minimum_new_user_bandwidth_mhz"]
        candidate[limits["key"]]["users"].append({
            "user_id": self.user_id, "cqi": user["cqi"], "bandwidth": b, "rate": rate_mbps(b, user["cqi"]),
            "latency": self.latency_heuristic(slice_type, user["request"], *limits["latency"]),
        })
        optimized = optimize_fixed_users(candidate, pinned_bandwidth={self.user_id: bandwidth} if bandwidth is not None else None)
        return self._save_plan("admission", context, optimized, slice_type, certificate)

    def plan_reallocation(self, slice_type=None):
        context = self._fresh_context()
        optimized = optimize_fixed_users(context["state"], [slice_type] if slice_type is not None else None)
        return self._save_plan("reallocation", context, optimized, slice_type)

    def _commit(self, plan_id, kind):
        if self.result is not None:
            raise ValueError("This request already has a terminal decision")
        if plan_id not in self.plans:
            raise ValueError("Unknown plan_id")
        plan = self.plans[plan_id]
        if plan["kind"] != kind:
            raise ValueError("Wrong commit tool for this plan type")
        if not plan["feasible"] or resource_violations(plan["candidate_state"]):
            raise ValueError("Infeasible plan cannot be committed")
        result = {"status": "admitted" if kind == "admission" else "reallocated",
                  "user_id": self.user_id, "slice_type": plan["slice_type"],
                  "plan_id": plan_id, "based_on_revision": plan["revision"],
                  "resource_engine": self.version, "changes": plan["changes"]}
        if kind == "admission":
            result["allocation"] = plan["allocation"]
        self.graph.apply_decision(plan["candidate_state"], result, plan["revision"])
        result["mapped_revision"] = self.graph.revision
        if kind == "admission":
            self.result = result
        return deepcopy(result)

    def commit_allocation(self, plan_id):
        return self._commit(plan_id, "admission")

    def commit_reallocation(self, plan_id):
        return self._commit(plan_id, "reallocation")

    def reject_request(self, slice_type, reason):
        context = self._fresh_context()
        certificate = bandwidth_options(context["state"], context["user"], slice_type)
        if certificate["admission_feasible"]:
            raise ValueError("Rejection denied: feasible bandwidth exists; call plan_allocation without bandwidth")
        result = {"status": "rejected", "user_id": self.user_id, "slice_type": slice_type,
                  "reason": str(reason), "feasibility_certificate": certificate,
                  "based_on_revision": context["revision"], "resource_engine": self.version}
        self.graph.apply_decision(context["state"], result, context["revision"])
        self.result = result | {"mapped_revision": self.graph.revision}
        return deepcopy(self.result)

    def get_history(self, revision=None):
        return {"events": self.graph.history()} if revision is None else {"revision": revision, **self.graph.state_at_revision(revision)}

    def execute_cypher(self, query):
        templates = {"current_state": self.graph.read_state, "schema": self.graph.schema, "recent_events": self.graph.history}
        if query not in templates:
            raise ValueError("Supported scoped Cypher templates: current_state, schema, recent_events")
        return {"query_template": query, "data": templates[query]()}

    def call(self, name, args, origin="model"):
        dispatch = {
            "get_decision_context": self.context, "search_entities": self.graph.search,
            "execute_cypher": self.execute_cypher, "Network_topology_query": self.graph.read_state,
            "get_bandwidth_options": self.get_bandwidth_options,
            "plan_allocation": self.plan_allocation, "commit_allocation": self.commit_allocation,
            "plan_reallocation": self.plan_reallocation, "commit_reallocation": self.commit_reallocation,
            "get_history": self.get_history, "reject_request": self.reject_request,
        }
        try:
            if name not in dispatch:
                raise ValueError("Unknown tool: " + str(name))
            if not isinstance(args, dict):
                raise ValueError("Tool args must be an object")
            forwarded = dict(args)
            full_evidence = forwarded.pop("full_evidence", False)
            if not isinstance(full_evidence, bool):
                raise ValueError("full_evidence must be boolean")
            result = dispatch[name](**forwarded)
        except (ValueError, TypeError) as exc:
            result = {"error": str(exc)}
        view = result if isinstance(args, dict) and args.get("full_evidence") is True else model_view(name, result)
        record = {"user_id": self.user_id, "tool": name, "args": args, "result": result,
                  "origin": origin, "model_observation": view,
                  "graph_revision": self.graph.revision, "tool_version": self.version}
        with (self.output / "tool_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result, json.dumps(view, ensure_ascii=False, separators=(",", ":"))
