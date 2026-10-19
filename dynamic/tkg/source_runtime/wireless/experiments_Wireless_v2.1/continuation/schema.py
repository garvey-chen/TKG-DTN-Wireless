"""Explicit metagraph for frame-synchronized wireless service observations.

This extends the v2 entity vocabulary. Simulation time, transaction revision,
and processing wall time remain separate; none is inferred from another.
"""

NODE_TYPES = {
    "Scenario": "Independent coordinate system and radio resource pools for one scene.",
    "BaseStation": "Logical radio resource provider associated with observed users.",
    "Slice": "eMBB or URLLC pool with the original v2 resource and QoS constraints.",
    "User": "Persistent scene/user identity and latest observed state; admission is explicit.",
    "ServiceRequirement": "Configured service contract for this monitoring experiment; no inferred intent claim.",
    "Allocation": "Versioned resource assignment; telemetry does not itself allocate resources.",
    "Frame": "Atomic scene snapshot at a relative simulation timestamp.",
    "TimeWindow": "Grouping of frames for temporal retrieval; not a second source of state.",
    "Observation": "Immutable per-user position and channel telemetry for a frame.",
    "ServiceStatus": "Per-observation evaluation under the allocation active at observation time.",
    "StateEvent": "Ordered state transition with revision and replayable state evidence.",
    "ServiceEvent": "Observed service/channel change with current and historical evidence.",
    "ServiceEpisode": "A service degradation interval, optionally closed by recovery.",
    "SchemaType": "Entity type in this explicit metagraph.",
}

RELATION_TYPES = {
    "HAS_BASE_STATION": ("Scenario", "BaseStation"),
    "PROVIDES": ("BaseStation", "Slice"),
    "SERVED_BY": ("User", "BaseStation"),
    "HAS_REQUIREMENT": ("User", "ServiceRequirement"),
    "REQUIRES_SLICE": ("ServiceRequirement", "Slice"),
    "HAS_ALLOCATION": ("User", "Allocation"),
    "CURRENT_ALLOCATION": ("User", "Allocation"),
    "ON_SLICE": ("Allocation", "Slice"),
    "HAS_WINDOW": ("Scenario", "TimeWindow"),
    "CONTAINS_FRAME": ("TimeWindow", "Frame"),
    "NEXT_FRAME": ("Frame", "Frame"),
    "HAS_OBSERVATION": ("User", "Observation"),
    "IN_FRAME": ("Observation", "Frame"),
    "NEXT_OBSERVATION": ("Observation", "Observation"),
    "RECORDED_IN": ("Observation", "StateEvent"),
    "HAS_SERVICE_STATUS": ("Observation", "ServiceStatus"),
    "EVALUATES_ALLOCATION": ("ServiceStatus", "Allocation"),
    "HAS_EVENT": ("Frame", "ServiceEvent"),
    "AFFECTS_USER": ("ServiceEvent", "User"),
    "EVIDENCED_BY": ("ServiceEvent", "Observation"),
    "IN_EPISODE": ("ServiceEvent", "ServiceEpisode"),
    "HAS_EPISODE": ("User", "ServiceEpisode"),
    "FOLLOWS": ("StateEvent", "StateEvent"),
    "INSTANCE_OF": ("*", "SchemaType"),
    "SCHEMA_RELATION": ("SchemaType", "SchemaType"),
}


def definition():
    """Return a JSON-serializable schema without depending on a live database."""
    return {
        "schema_version": "wireless_dynamic_v2.1",
        "node_types": dict(NODE_TYPES),
        "relationships": dict(RELATION_TYPES),
        "scope": "Entities and edges belong to run_id; user identity is (scenario, RX_ID).",
        "time": {
            "simulation_time": "Relative time_s, supplied by mobility frames, in seconds.",
            "revision": "Monotonically increasing committed graph version; not seconds.",
            "observed_at": "Optional UTC processing wall time; not simulated event time.",
            "next_edges": "NEXT_FRAME and NEXT_OBSERVATION point from earlier to later.",
            "follows_edge": "FOLLOWS points from a newer StateEvent to its predecessor.",
        },
        "semantics": {
            "SERVED_BY": "Physical/logical association only; inspect admission and allocation before counting service.",
            "EVALUATES_ALLOCATION": "Present only for admitted users with an active allocation.",
            "observations": "One immutable observation per observed user and frame, including stationary users.",
            "state_vs_history": "Current User/Allocation projections coexist with immutable telemetry and service history.",
        },
        "limitations": [
            "Rates use the existing v2 CQI/bandwidth engineering formula, not a standardized PHY model.",
            "Assigned latency is a v2 heuristic, not measured end-to-end delay.",
            "Service contracts are configured before monitoring; this stage does not evaluate agent intent inference.",
            "Offline graph exports do not imply Neo4j import or autonomous agent execution.",
        ],
    }


def mermaid():
    """Render the actual entity relations, without duplicating instance typing."""
    lines = ["graph LR"]
    for relation, (source, target) in RELATION_TYPES.items():
        if relation in ("INSTANCE_OF", "SCHEMA_RELATION"):
            continue
        lines.append(f"  {source} -->|{relation}| {target}")
    lines.append("  User -. INSTANCE_OF .-> SchemaType")
    lines.append("  SchemaType -. SCHEMA_RELATION .-> SchemaType")
    return "\n".join(lines) + "\n"
