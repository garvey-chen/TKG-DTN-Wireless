"""Graph-grounded tools and versioned network actions for the copied ReAct agent."""

import json
import uuid
from copy import deepcopy

from ..metrics import SLICE_LIMITS, rate_mbps
from ..policies import load_legacy
from .executor import resource_violations
from .graph_store import normalized


TOOL_HELP = """
get_decision_context(): Read the current user, current slice allocation graph, constraints and
  the SAME complete intent knowledge text used by WirelessAgent. Required before decisions.
search_entities(keyword: str or list): Search arrived users, slices and schema types in this run.
execute_cypher(query: str): Execute a scoped read template: 'current_state', 'schema', or 'recent_events'.
  Free-form Cypher is deliberately unavailable to the model in this prototype; templates are logged.
Network_topology_query(): Read the current user-slice-resource projection from Neo4j.
plan_allocation(slice_type: 'eMBB'|'URLLC', bandwidth: optional integer): Dry run. Without bandwidth,
  call original WirelessAgent beamforming_tool with the same LLM and CQI. With bandwidth, use that
  integer directly. In both cases reuse original latency heuristic and original capacity/reduction
  functions. Returns plan_id, validity, changes and graph revision. Does NOT change the network.
commit_allocation(plan_id: str): Atomically apply an existing valid, fresh plan and map it to Neo4j.
reject_request(slice_type: 'eMBB'|'URLLC', reason: str): Reject when no suitable plan works.
  Requires at least one failed plan for this request. Records the decision in the graph.
finish(): Finish only after a committed allocation or recorded rejection.
"""


class WirelessTools:
    def __init__(self, graph, llm, output):
        self.graph, self.llm, self.output = graph, llm, output
        self.legacy = load_legacy("agent_kb")
        self.legacy.llm = llm
        self.user_id = None
        self.plans = {}
        self.result = None
        self.context_read = False
        self.failed_plans = 0

    def start(self, user_id):
        self.user_id = str(user_id)
        self.plans, self.result, self.context_read, self.failed_plans = {}, None, False, 0

    def context(self):
        value = self.graph.context(self.user_id)
        self.context_read = True
        return value

    def plan_allocation(self, slice_type, bandwidth=None):
        if not self.context_read:
            raise ValueError("Read get_decision_context before planning")
        if slice_type not in SLICE_LIMITS:
            raise ValueError("Unknown slice")
        context = self.graph.context(self.user_id)
        before = deepcopy(context["state"])
        if any(str(u["user_id"]) == self.user_id for v in SLICE_LIMITS.values() for u in before[v["key"]]["users"]):
            raise ValueError("Current user is already allocated")
        self.legacy.update_network_state(deepcopy(before))
        user, limits = context["user"], SLICE_LIMITS[slice_type]
        failures_before = self.llm.failures
        if bandwidth is None:
            proposed = self.legacy.beamforming_tool.invoke({"user_id": self.user_id, "slice_type": slice_type,
                                                          "cqi": user["cqi"], "request": user["request"]})
            if self.llm.failures > failures_before:
                raise RuntimeError("Original resource tool caught an API error; fallback discarded")
            bandwidth = proposed["allocated_bandwidth"]
            latency = proposed["allocated_latency"]
        else:
            if isinstance(bandwidth, bool) or not isinstance(bandwidth, int):
                raise ValueError("Bandwidth must be an integer number of MHz")
            latency = self.legacy.apply_heuristic_latency(slice_type, user["request"], *limits["latency"])
            proposed = {"source": "agent_selected_integer", "allocated_bandwidth": bandwidth}
        capacity = self.legacy.check_and_adjust_capacity.invoke({"slice_type": slice_type, "required_bandwidth": bandwidth})
        candidate = deepcopy(self.legacy.get_current_network_state())
        allocation = {"user_id": self.user_id, "bandwidth": bandwidth, "cqi": user["cqi"],
                      "rate": rate_mbps(bandwidth, user["cqi"]), "latency": latency}
        candidate[limits["key"]]["users"].append(allocation)
        candidate = normalized(candidate)
        violations = resource_violations(candidate)
        if not capacity["has_capacity"]:
            violations.append("insufficient_capacity")
        plan_id = uuid.uuid4().hex[:12]
        plan = {"plan_id": plan_id, "revision": context["revision"], "slice_type": slice_type,
                "feasible": not violations, "violations": violations, "allocation": allocation,
                "capacity_adjustment": capacity, "resource_proposal": proposed, "candidate_state": candidate}
        self.plans[plan_id] = plan
        self.failed_plans += int(bool(violations))
        # Legacy state is a sandbox for each plan, never the authoritative twin.
        self.legacy.update_network_state(before)
        return plan

    def commit_allocation(self, plan_id):
        if self.result is not None:
            raise ValueError("This request already has a terminal decision")
        if plan_id not in self.plans:
            raise ValueError("Unknown plan_id")
        plan = self.plans[plan_id]
        if not plan["feasible"] or resource_violations(plan["candidate_state"]):
            raise ValueError("Infeasible plan cannot be committed")
        result = {"status": "admitted", "user_id": self.user_id, "slice_type": plan["slice_type"],
                  "allocation": plan["allocation"], "plan_id": plan_id,
                  "based_on_revision": plan["revision"], "capacity_adjustment": plan["capacity_adjustment"]}
        self.graph.apply_decision(plan["candidate_state"], result, plan["revision"])
        self.result = result | {"mapped_revision": self.graph.revision}
        return self.result

    def reject_request(self, slice_type, reason):
        if self.result is not None or slice_type not in SLICE_LIMITS or not self.failed_plans:
            raise ValueError("Rejection requires a failed plan and no existing terminal decision")
        result = {"status": "rejected", "user_id": self.user_id, "slice_type": slice_type,
                  "reason": str(reason), "based_on_revision": self.graph.revision}
        self.graph.apply_decision(self.graph.read_state(), result, self.graph.revision)
        self.result = result | {"mapped_revision": self.graph.revision}
        return self.result

    def execute_cypher(self, query):
        templates = {"current_state": self.graph.read_state, "schema": self.graph.schema, "recent_events": self.graph.history}
        if query not in templates:
            raise ValueError("Supported scoped Cypher templates: current_state, schema, recent_events")
        return {"query_template": query, "data": templates[query]()}

    def call(self, name, args):
        dispatch = {
            "get_decision_context": self.context,
            "search_entities": self.graph.search,
            "execute_cypher": self.execute_cypher,
            "Network_topology_query": self.graph.read_state,
            "plan_allocation": self.plan_allocation,
            "commit_allocation": self.commit_allocation,
            "reject_request": self.reject_request,
        }
        try:
            if name not in dispatch:
                raise ValueError("Unknown tool: " + str(name))
            if not isinstance(args, dict):
                raise ValueError("Tool args must be an object")
            result = dispatch[name](**args)
        except (ValueError, TypeError) as exc:
            # Correctable tool misuse returns evidence to ReAct. Infrastructure
            # failures propagate and mark the trial failed, never as a rejection.
            result = {"error": str(exc)}
        record = {"user_id": self.user_id, "tool": name, "args": args, "result": result,
                  "graph_revision": self.graph.revision}
        with (self.output / "tool_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result, json.dumps(result, ensure_ascii=False)
