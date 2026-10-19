"""Event-sourced wireless digital twin backed by a real, run-scoped Neo4j graph.

The simulator owns physical state. Mapping commits observations + projections in
one transaction; agent tools read those projections back through Cypher. Full
event snapshots support replay without exposing future observations to an agent.
"""

import json
import math
import uuid
from copy import deepcopy
from datetime import datetime, timezone

from .. import ROOT
from ..metrics import SLICE_LIMITS, empty_state, rate_mbps
from .schema import NODE_TYPES, RELATION_TYPES, definition, mermaid


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def normalized(state):
    result = empty_state()
    for limits in SLICE_LIMITS.values():
        part = result[limits["key"]]
        part["users"] = sorted(deepcopy(state[limits["key"]]["users"]), key=lambda u: str(u["user_id"]))
        part["resource_usage"] = sum(u["bandwidth"] for u in part["users"])
        part["utilization_rate"] = f'{part["resource_usage"] / part["total_capacity"] * 100:.2f}%'
    result["total_users"] = sum(len(result[v["key"]]["users"]) for v in SLICE_LIMITS.values())
    return result


class WirelessGraph:
    def __init__(self, connection, output, run_id=None):
        self.connection = connection
        self.output = output
        self.run_id = run_id or "wireless_" + uuid.uuid4().hex
        self.revision = 0
        self.logical_time = -1.0
        self.users = {}
        self.state = normalized(empty_state())
        self.output.mkdir(parents=True, exist_ok=True)
        self._write_query = self._mapping_query()

    def _mapping_query(self):
        labels = "\n".join(
            f"FOREACH (_ IN CASE WHEN row.kind = '{kind}' THEN [1] ELSE [] END | SET n:Wireless{kind})"
            for kind in NODE_TYPES
        )
        relations = "\n".join(
            f"FOREACH (_ IN CASE WHEN edge.type = '{name}' THEN [1] ELSE [] END | "
            f"MERGE (a)-[r:{name} {{key:edge.key}}]->(b) SET r += edge.props, r.run_id=$rid)"
            for name in (*RELATION_TYPES, "INSTANCE_OF", "SCHEMA_RELATION")
        )
        # The dependent SET takes a write lock before checking revision. All
        # following writes are inside the same transaction and gated by this row.
        return """
MATCH (run:WirelessRun {run_id:$rid})
SET run.lock_counter = coalesce(run.lock_counter,0) + 1
WITH run WHERE run.revision = $expected
SET run.revision=$revision, run.logical_time=$time
WITH run
CALL {
  WITH run
  MATCH (a:WirelessAllocation {run_id:$rid}) WHERE a.valid_to IS NULL
  SET a.valid_to=$revision
}
CALL {
  WITH run
  MATCH (:WirelessUser {run_id:$rid})-[r:CURRENT_ALLOCATION]->()
  DELETE r
}
CALL {
  WITH run
  UNWIND $nodes AS row
  MERGE (n:WirelessEntity {run_id:$rid, id:row.id})
  SET n += row.props, n.kind=row.kind
""" + labels + """
}
CALL {
  WITH run
  UNWIND $edges AS edge
  MATCH (a:WirelessEntity {run_id:$rid,id:edge.source}), (b:WirelessEntity {run_id:$rid,id:edge.target})
""" + relations + """
}
RETURN run.revision AS revision
"""

    def initialize(self):
        rows = self.connection.query(
            "MERGE (r:WirelessRun {run_id:$rid}) ON CREATE SET r.revision=0 RETURN r.revision AS revision",
            {"rid": self.run_id},
        )
        if rows != [{"revision": 0}]:
            raise RuntimeError("Refusing to overwrite an existing graph run")
        self._persist("initialize", 0, self.users, self.state, {"source": "static_scenario_adapter"}, initial=True)
        (self.output / "schema_definition.json").write_text(packed(definition()) + "\n", encoding="utf-8")
        (self.output / "schema.mmd").write_text(mermaid(), encoding="utf-8")
        (self.output / "mapping.cypher").write_text(self._write_query, encoding="utf-8")

    def _persist(self, event_type, logical_time, users, state, payload, initial=False):
        if not math.isfinite(float(logical_time)) or logical_time < self.logical_time:
            raise ValueError("Out-of-order/nonfinite logical time; event not applied")
        revision = self.revision + 1
        nodes, edges = [], []

        def node(kind, identifier, **props):
            nodes.append({"id": identifier, "kind": kind, "props": props})
            if kind != "SchemaType":
                edge("INSTANCE_OF", identifier, "type:" + kind)

        def edge(relation, source, target, **props):
            edges.append({"type": relation, "source": source, "target": target,
                          "key": relation + ":" + source + ":" + target, "props": props})

        if initial:
            for kind, description in NODE_TYPES.items():
                node("SchemaType", "type:" + kind, name=kind, description=description)
            for relation, (source, target) in RELATION_TYPES.items():
                edge("SCHEMA_RELATION", "type:" + source, "type:" + target, name=relation)
                # Parallel schema relations between the same types need distinct keys.
                edges[-1]["key"] += ":" + relation
            node("BaseStation", "bs:1", name="WirelessAgent logical base station")
            text = (ROOT / "data/knowledge/Intent_Understand.txt").read_text(encoding="utf-8")
            node("Knowledge", "knowledge:intent", text=text, source="same upstream Intent_Understand.txt")
            for name in SLICE_LIMITS:
                edge("PROVIDES", "bs:1", "slice:" + name)
                edge("GUIDES", "knowledge:intent", "slice:" + name)
        for name, limits in SLICE_LIMITS.items():
            part = state[limits["key"]]
            node("Slice", "slice:" + name, name=name, capacity=limits["capacity"],
                 bandwidth_min=limits["bandwidth"][0], bandwidth_max=limits["bandwidth"][1],
                 rate_min=limits["rate"][0], rate_max=limits["rate"][1],
                 latency_min=limits["latency"][0], latency_max=limits["latency"][1],
                 resource_usage=part["resource_usage"], revision=revision)
            for allocation in part["users"]:
                uid = str(allocation["user_id"])
                aid = f"allocation:{uid}:{revision}"
                node("Allocation", aid, **allocation, slice_type=name, valid_from=revision, valid_to=None)
                edge("HAS_ALLOCATION", "user:" + uid, aid)
                edge("CURRENT_ALLOCATION", "user:" + uid, aid)
                edge("ON_SLICE", aid, "slice:" + name)
        for uid, observation in users.items():
            node("User", "user:" + uid, user_id=uid, request=observation["request"],
                 location_json=packed(observation["location"]), cqi=observation["cqi"],
                 present=observation.get("present", True), revision=revision)
            edge("SERVED_BY", "user:" + uid, "bs:1")
        eid = f"event:{revision}"
        node("StateEvent", eid, event_type=event_type, revision=revision, logical_time=logical_time,
             observed_at=datetime.now(timezone.utc).isoformat(), payload_json=packed(payload),
             state_json=packed(state), users_json=packed(users))
        if self.revision:
            edge("FOLLOWS", eid, f"event:{self.revision}")
        uid = payload.get("user_id")
        if uid is not None:
            oid = f"observation:{uid}:{revision}"
            node("Observation", oid, user_id=str(uid), logical_time=logical_time, revision=revision,
                 event_type=event_type, payload_json=packed(payload))
            edge("HAS_OBSERVATION", "user:" + str(uid), oid)
            edge("RECORDED_IN", oid, eid)
        params = {"rid": self.run_id, "expected": self.revision, "revision": revision,
                  "time": logical_time, "nodes": nodes, "edges": edges}
        rows = self.connection.query(self._write_query, params)
        if rows != [{"revision": revision}]:
            raise RuntimeError("Stale graph revision; mapping not committed")
        self.revision, self.logical_time = revision, logical_time
        self.users, self.state = deepcopy(users), deepcopy(state)
        # Read the actual graph projection; never accept a local-only 'sync'.
        projected = self.read_state()
        if normalized(projected) != normalized(state):
            raise RuntimeError("Neo4j projection differs from simulator state")
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(packed({"run_id": self.run_id, "revision": revision, "event_type": event_type,
                                 "logical_time": logical_time, "payload": payload, "state": state,
                                 "projection_verified": True}) + "\n")

    def arrive(self, observation, logical_time):
        required = {"user_id", "location", "request", "cqi"}
        if set(observation) != required:
            raise ValueError("Observation must contain only user_id, location, request, cqi; evaluator labels are forbidden")
        uid = str(observation["user_id"])
        if uid in self.users:
            raise ValueError("Duplicate arrival")
        if not 1 <= int(observation["cqi"]) <= 15:
            raise ValueError("CQI must be 1..15")
        users = deepcopy(self.users)
        users[uid] = deepcopy(observation) | {"user_id": uid, "present": True}
        self._persist("arrival", logical_time, users, self.state, deepcopy(observation))

    def apply_decision(self, state, result, expected_revision):
        if expected_revision != self.revision:
            raise ValueError("Stale allocation plan; obtain a fresh graph observation")
        self._persist("decision", self.logical_time, self.users, normalized(state), result)

    def update_channel(self, user_id, cqi, logical_time):
        """Reserved dynamic-event entry point; no autonomous future CQI generation."""
        uid = str(user_id)
        if uid not in self.users or not self.users[uid]["present"] or not 1 <= int(cqi) <= 15:
            raise ValueError("Unknown/departed user or invalid CQI")
        users, state = deepcopy(self.users), deepcopy(self.state)
        users[uid]["cqi"] = int(cqi)
        for limits in SLICE_LIMITS.values():
            for allocation in state[limits["key"]]["users"]:
                if str(allocation["user_id"]) == uid:
                    allocation["cqi"] = int(cqi)
                    allocation["rate"] = rate_mbps(allocation["bandwidth"], int(cqi))
        self._persist("channel_update", logical_time, users, normalized(state), {"user_id": uid, "cqi": int(cqi)})

    def depart(self, user_id, logical_time):
        uid = str(user_id)
        if uid not in self.users or not self.users[uid]["present"]:
            raise ValueError("Unknown/departed user")
        users, state = deepcopy(self.users), deepcopy(self.state)
        users[uid]["present"] = False
        for limits in SLICE_LIMITS.values():
            part = state[limits["key"]]
            part["users"] = [u for u in part["users"] if str(u["user_id"]) != uid]
        self._persist("departure", logical_time, users, normalized(state), {"user_id": uid})

    def read_state(self):
        rows = self.connection.query("""
MATCH (s:WirelessSlice {run_id:$rid})
OPTIONAL MATCH (u:WirelessUser {run_id:$rid})-[:CURRENT_ALLOCATION]->(a:WirelessAllocation {run_id:$rid})-[:ON_SLICE]->(s)
RETURN s.name AS slice_type, s.resource_usage AS usage, collect(properties(a)) AS allocations
""", {"rid": self.run_id})
        if {r["slice_type"] for r in rows} != set(SLICE_LIMITS):
            raise RuntimeError("Missing slice projection in graph")
        state = empty_state()
        for row in rows:
            part = state[SLICE_LIMITS[row["slice_type"]]["key"]]
            part["users"] = [{k: a[k] for k in ("user_id", "cqi", "bandwidth", "rate", "latency")} for a in row["allocations"]]
            if abs(row["usage"] - sum(a["bandwidth"] for a in part["users"])) > 1e-6:
                raise RuntimeError("Slice accounting differs from graph allocation edges")
        return normalized(state)

    def context(self, user_id):
        rows = self.connection.query("""
MATCH (r:WirelessRun {run_id:$rid}), (u:WirelessUser {run_id:$rid,user_id:$uid}), (k:WirelessKnowledge {run_id:$rid})
RETURN r.revision AS revision, properties(u) AS user, k.text AS knowledge
""", {"rid": self.run_id, "uid": str(user_id)})
        if len(rows) != 1 or not rows[0]["user"]["present"]:
            raise ValueError("User is not currently present in this run")
        return rows[0] | {"state": self.read_state(), "constraints": SLICE_LIMITS}

    def schema(self):
        return self.connection.query("""
MATCH (a:WirelessSchemaType {run_id:$rid})
OPTIONAL MATCH (a)-[r:SCHEMA_RELATION]->(b:WirelessSchemaType {run_id:$rid})
RETURN a.name AS source_type, a.description AS description, collect({relation:r.name,target:b.name}) AS relations
ORDER BY source_type
""", {"rid": self.run_id})

    def history(self, limit=10):
        return self.connection.query("""
MATCH (e:WirelessStateEvent {run_id:$rid}) RETURN e.revision AS revision, e.logical_time AS logical_time,
e.event_type AS event_type, e.payload_json AS payload_json ORDER BY revision DESC LIMIT $limit
""", {"rid": self.run_id, "limit": min(50, max(1, int(limit)))})

    def search(self, keyword, limit=15):
        words = keyword if isinstance(keyword, list) else [keyword]
        return self.connection.query("""
MATCH (n:WirelessEntity {run_id:$rid}) WHERE n.kind IN ['User','Slice','BaseStation','SchemaType']
AND any(word IN $words WHERE toLower(n.id + ' ' + coalesce(n.request,n.name,'')) CONTAINS toLower(word))
RETURN properties(n) AS entity LIMIT $limit
""", {"rid": self.run_id, "words": [str(w) for w in words], "limit": min(30, int(limit))})

    def export(self):
        nodes = self.connection.query("MATCH (n:WirelessEntity {run_id:$rid}) RETURN properties(n) AS node", {"rid": self.run_id})
        edges = self.connection.query("""
MATCH (a:WirelessEntity {run_id:$rid})-[r]->(b:WirelessEntity {run_id:$rid})
RETURN a.id AS source, type(r) AS type, b.id AS target, properties(r) AS properties
""", {"rid": self.run_id})
        (self.output / "graph_snapshot.json").write_text(packed({"run_id": self.run_id, "revision": self.revision,
            "nodes": [r["node"] for r in nodes], "edges": edges}), encoding="utf-8")
        (self.output / "schema_graph.json").write_text(packed(self.schema()), encoding="utf-8")
        (self.output / "connection.json").write_text(packed(self.connection.public_settings() | {"run_id": self.run_id,
            "transactions": self.connection.calls, "transport_seconds": self.connection.seconds}), encoding="utf-8")
