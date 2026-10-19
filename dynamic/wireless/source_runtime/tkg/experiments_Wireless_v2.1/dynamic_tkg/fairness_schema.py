"""Additive schema for causally accumulated modeled traffic history."""

from .schema import NODE_TYPES as BASE_NODES, RELATION_TYPES as BASE_RELATIONS
from .schema import definition as base_definition

NODE_TYPES = dict(BASE_NODES, TrafficInterval=(
    "Closed simulated service interval: previous post-control rate held until the next observation; not measured traffic."))
RELATION_TYPES = dict(BASE_RELATIONS, **{
    "HAS_TRAFFIC_INTERVAL": ("User", "TrafficInterval"),
    "TRAFFIC_FROM_OBSERVATION": ("TrafficInterval", "Observation"),
    "TRAFFIC_UNDER_ALLOCATION": ("TrafficInterval", "Allocation"),
    "TRAFFIC_FROM_STATUS": ("TrafficInterval", "ServiceStatus"),
    "TRAFFIC_START_FRAME": ("TrafficInterval", "Frame"),
    "TRAFFIC_END_FRAME": ("TrafficInterval", "Frame"),
    "TRAFFIC_RECORDED_IN": ("TrafficInterval", "StateEvent"),
})


def definition():
    value = base_definition()
    value.update(schema_version="wireless_dynamic_v2.1_long_term_fairness",
                 node_types=dict(NODE_TYPES), relationships=dict(RELATION_TYPES))
    value["semantics"]["traffic"] = (
        "For each admitted user, completed [previous_time,current_time) carries the prior post-control "
        "modeled rate times elapsed seconds in Mbit. Controls at the same timestamp add no traffic. "
        "No interval beyond the final observed timestamp is invented; observed-only users are excluded.")
    return value


def mermaid():
    return "graph LR\n" + "\n".join(
        f"  {source} -->|{relation}| {target}"
        for relation, (source, target) in RELATION_TYPES.items()
        if relation not in ("INSTANCE_OF", "SCHEMA_RELATION")) + "\n"
