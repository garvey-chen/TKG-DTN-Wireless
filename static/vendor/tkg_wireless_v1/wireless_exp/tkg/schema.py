"""Explicit schema metagraph, independent of the current instance population."""

NODE_TYPES = {
    "BaseStation": "Single logical radio resource provider; no new propagation simulation.",
    "Slice": "eMBB/URLLC bandwidth pool and the original QoS constraints.",
    "User": "Only an arrived user; latest request, location, CQI and presence.",
    "Observation": "Immutable arrival/channel/departure telemetry with event time and revision.",
    "Allocation": "Resource assignment version; valid_from inclusive, valid_to exclusive (revisions).",
    "StateEvent": "Immutable event, simulator snapshot, source and logical time for replay.",
    "Knowledge": "Exact same public intent knowledge text as WirelessAgent; no evaluator labels.",
    "SchemaType": "Entity-type definition (schema graph).",
}

RELATION_TYPES = {
    "PROVIDES": ("BaseStation", "Slice"),
    "SERVED_BY": ("User", "BaseStation"),
    "HAS_OBSERVATION": ("User", "Observation"),
    "CURRENT_ALLOCATION": ("User", "Allocation"),
    "HAS_ALLOCATION": ("User", "Allocation"),
    "ON_SLICE": ("Allocation", "Slice"),
    "RECORDED_IN": ("Observation", "StateEvent"),
    "FOLLOWS": ("StateEvent", "StateEvent"),
    "GUIDES": ("Knowledge", "Slice"),
}


def definition():
    return {
        "node_types": NODE_TYPES, "relationships": RELATION_TYPES,
        "scope": "Every node/relationship belongs to one run_id. SchemaType nodes and SCHEMA_RELATION edges are materialized in Neo4j.",
        "time": "revision is the monotonically increasing graph version; logical_time is supplied by the event source; observed_at is UTC wall clock.",
        "limitations": "Static radio measurements; assigned latency is an upstream heuristic, not measured end-to-end delay. No trained temporal embeddings in this adapter.",
    }


def mermaid():
    lines = ["graph LR"]
    lines.extend(f"  {a} -->|{relation}| {b}" for relation, (a, b) in RELATION_TYPES.items())
    lines.append("  User -. INSTANCE_OF .-> SchemaType")
    return "\n".join(lines) + "\n"
