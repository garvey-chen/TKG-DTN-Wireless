"""Causal modeled traffic ledger and history-aware allocation transactions.

This extension leaves all preceding experimental graph/control code unchanged.
Traffic is a model-derived integral, not measured packet-counter telemetry.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3

from .control import ControlledWirelessGraph
from .graph import packed
from .service import evaluate_service, _jain
from .fairness_schema import NODE_TYPES, RELATION_TYPES, definition


POLICY = "fixed_cohort_max_effective_then_historical_proportional_fairness"


class FairnessWirelessGraph(ControlledWirelessGraph):
    def __init__(self, database, run_id=None, *, static_terminal_dir, history_window_s=None, horizon_s=10.0):
        if history_window_s is not None and (isinstance(history_window_s, bool) or
                not math.isfinite(float(history_window_s)) or history_window_s <= 0):
            raise ValueError("history_window_s must be positive or None for cumulative history")
        if isinstance(horizon_s, bool) or not math.isfinite(float(horizon_s)) or horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        if history_window_s is not None and history_window_s < horizon_s:
            raise ValueError("history_window_s must be at least the planning horizon")
        self.history_window_s, self.horizon_s = history_window_s, float(horizon_s)
        config = {"history_window_s": history_window_s, "horizon_s": self.horizon_s,
                  "traffic_source": "modeled_previous_post_control_rate_integral"}
        # Opening an old result must not silently migrate or change its hashes.
        path = Path(database)
        if path.exists() and path.stat().st_size:
            old = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                row = old.execute("SELECT value_json FROM metadata WHERE key='fairness_config'").fetchone()
                if row is None:
                    raise ValueError("Use a fresh database; existing results are not migrated to fairness mode")
                if json.loads(row[0]) != config:
                    raise ValueError("Fairness configuration differs from the existing run")
            finally:
                old.close()
        super().__init__(database, run_id, static_terminal_dir=static_terminal_dir)
        self.db.executescript("""
CREATE TABLE IF NOT EXISTS traffic_intervals(
 scenario TEXT NOT NULL, user_id TEXT NOT NULL,
 start_time_s REAL NOT NULL, end_time_s REAL NOT NULL,
 source_revision INTEGER NOT NULL, recorded_revision INTEGER NOT NULL,
 properties_json TEXT NOT NULL,
 PRIMARY KEY(scenario,user_id,end_time_s));
CREATE INDEX IF NOT EXISTS traffic_by_time ON traffic_intervals(scenario,end_time_s,user_id);
""")
        if self.db.execute("SELECT 1 FROM metadata WHERE key='fairness_config'").fetchone() is None:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.execute("INSERT INTO metadata VALUES(?,?)", ("fairness_config", packed(config)))
                self.db.execute("UPDATE metadata SET value_json=? WHERE key='schema'", (packed(definition()),))
                self._node("SchemaType", "schema:TrafficInterval", "", name="TrafficInterval",
                           description=NODE_TYPES["TrafficInterval"])
                for relation, (source, target) in RELATION_TYPES.items():
                    if "TrafficInterval" in (source, target):
                        self._edge("SCHEMA_RELATION", "schema:" + source, "schema:" + target, "",
                                   key_suffix=relation, relation_name=relation)
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                self.close()
                raise

    def _node(self, kind, identifier, scenario, **props):
        if kind != "TrafficInterval":
            return super()._node(kind, identifier, scenario, **props)
        props = dict(props, run_id=self.run_id, scenario=scenario)
        # Intervals are immutable evidence, never upserted or overwritten.
        self.db.execute("INSERT INTO nodes VALUES(?,?,?,?)", (identifier, kind, scenario, packed(props)))
        self._edge("INSTANCE_OF", identifier, "schema:" + kind, scenario)

    def _edge(self, relation, source, target, scenario, key_suffix="", **props):
        if relation not in RELATION_TYPES:
            raise ValueError(f"Unknown relationship: {relation}")
        identifier = packed([source, relation, target, key_suffix])
        self.db.execute("INSERT INTO edges VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET properties_json=excluded.properties_json",
                        (identifier, relation, scenario, source, target, packed(dict(props, run_id=self.run_id))))

    def _materialize(self, state, previous, events):
        state["fairness_policy"] = POLICY
        state["fairness_history_window_s"] = self.history_window_s
        super()._materialize(state, previous, events)
        if previous is None:
            return
        scene = state["scenario"]
        start, end = previous["time_s"], state["time_s"]
        duration = end - start
        for uid in previous["allocations"]:
            service = previous["users"][uid]["service"]
            rate, minimum = service["rate_mbps"], service["rate_min_mbps"]
            aid = self._active_allocation_id(scene, uid)
            start_frame = previous["frame_index"]
            oid = f"{scene}:observation:{uid}:{start_frame:04d}"
            status = f"{scene}:service_status:{uid}:{start_frame:04d}"
            if previous.get("phase") == "post_control":
                status += f":control:{previous['revision']}"
            interval_id = f"{scene}:traffic:{uid}:{start_frame:04d}"
            props = {
                "interval_id": interval_id, "user_id": uid,
                "start_time_s": start, "end_time_s": end, "duration_s": duration,
                "start_frame_index": start_frame, "end_frame_index": state["frame_index"],
                "source_revision": previous["revision"], "recorded_revision": state["revision"],
                "source_phase": previous["phase"], "source_observation_id": oid,
                "source_allocation_id": aid, "source_service_status_id": status,
                "slice_type": service["slice_type"], "modeled_rate_mbps": rate,
                "minimum_rate_mbps": minimum, "service_effective": service["effective"],
                "traffic_mbit": rate * duration, "served_mbit": rate * duration,
                "effective_served_mbit": rate * duration if service["effective"] else 0.0,
                "demand_mbit": minimum * duration,
                "deficit_mbit": max(0.0, minimum - rate) * duration,
                "effective_time_s": duration if service["effective"] else 0.0,
                "source": "modeled_previous_post_control_rate_integral",
                "traffic_assumption": "saturated/backlogged service under the existing modeled rate",
                "measured_packet_traffic": False,
            }
            self.db.execute("INSERT INTO traffic_intervals VALUES(?,?,?,?,?,?,?)",
                            (scene, uid, start, end, previous["revision"], state["revision"], packed(props)))
            self._node("TrafficInterval", interval_id, scene, **props)
            self._edge("HAS_TRAFFIC_INTERVAL", f"{scene}:user:{uid}", interval_id, scene)
            for relation, target in (
                ("TRAFFIC_FROM_OBSERVATION", oid), ("TRAFFIC_UNDER_ALLOCATION", aid),
                ("TRAFFIC_FROM_STATUS", status),
                ("TRAFFIC_START_FRAME", f"{scene}:frame:{start_frame:04d}"),
                ("TRAFFIC_END_FRAME", f"{scene}:frame:{state['frame_index']:04d}"),
                ("TRAFFIC_RECORDED_IN", f"{scene}:state_event:{state['revision']:04d}"),
            ):
                self._edge(relation, interval_id, target, scene)

    def traffic_history(self, scene, *, through_time_s=None, window_s=None, user_id=None,
                        through_revision=None):
        """Return only closed intervals known by the requested time/revision.

        A trailing window may clip an already closed interval. Querying at an
        unobserved past timestamp never reveals an interval completed later.
        ``window_s=None`` explicitly means cumulative history.
        """
        current = self.current_state(scene)
        cutoff = current["time_s"] if through_time_s is None else through_time_s
        revision = current["revision"] if through_revision is None else through_revision
        if isinstance(cutoff, bool) or not math.isfinite(float(cutoff)) or not 0 <= cutoff <= current["time_s"]:
            raise ValueError("Traffic history cutoff must be between t=0 and current observed time")
        if isinstance(revision, bool) or not isinstance(revision, int) or not 1 <= revision <= current["revision"]:
            raise ValueError("Traffic history revision must be a committed positive revision")
        if window_s is not None and (isinstance(window_s, bool) or not math.isfinite(float(window_s)) or window_s <= 0):
            raise ValueError("window_s must be positive or None")
        start = max(0.0, float(cutoff) - window_s) if window_s is not None else 0.0
        ids = list(current["allocations"])
        if user_id is not None:
            uid = str(user_id)
            if uid not in current["users"]:
                raise ValueError("Unknown user")
            ids = [uid] if uid in current["allocations"] else []
        users = {}
        for uid in ids:
            contract = current["contracts"][uid]
            users[uid] = {"user_id": uid, "slice_type": contract["slice_type"],
                          "min_rate_mbps": contract["min_rate_mbps"],
                          **{k: 0.0 for k in ("traffic_mbit", "served_mbit", "effective_served_mbit", "demand_mbit",
                                               "deficit_mbit", "effective_time_s", "observed_duration_s")}}
        records = self.db.execute(
            "SELECT properties_json FROM traffic_intervals WHERE scenario=? AND end_time_s<=? "
            "AND end_time_s>? AND recorded_revision<=? ORDER BY end_time_s,user_id",
            (scene, cutoff, start, revision)).fetchall()
        intervals = []
        for row in records:
            props = json.loads(row[0])
            uid = props["user_id"]
            if uid not in users:
                continue
            overlap_start = max(start, props["start_time_s"])
            overlap_end = min(float(cutoff), props["end_time_s"])
            elapsed = overlap_end - overlap_start
            if elapsed <= 0:
                continue
            fraction = elapsed / props["duration_s"]
            for key in ("traffic_mbit", "served_mbit", "effective_served_mbit", "demand_mbit", "deficit_mbit", "effective_time_s"):
                users[uid][key] += props[key] * fraction
            users[uid]["observed_duration_s"] += elapsed
            intervals.append(dict(props, overlap_start_time_s=overlap_start, overlap_end_time_s=overlap_end,
                                  included_duration_s=elapsed, included_traffic_mbit=props["traffic_mbit"] * fraction))
        for value in users.values():
            duration = value["observed_duration_s"]
            value["mean_rate_mbps"] = value["traffic_mbit"] / duration if duration else None
            value["normalized_mean_rate"] = value["mean_rate_mbps"] / value["min_rate_mbps"] if duration else None
            value["effective_time_fraction"] = value["effective_time_s"] / duration if duration else None
        durations = {round(v["observed_duration_s"], 10) for v in users.values()}
        if len(durations) > 1:
            raise RuntimeError("Incomplete fixed-cohort traffic evidence")
        duration = next(iter(durations), 0.0)
        normalized = [v["normalized_mean_rate"] for v in users.values()]
        rates = [v["mean_rate_mbps"] for v in users.values()]
        metrics = {
            "jain_cumulative_rate_all_admitted": _jain(rates) if duration else None,
            "jain_cumulative_demand_normalized_all_admitted": _jain(normalized) if duration else None,
            "total_traffic_mbit": sum(v["traffic_mbit"] for v in users.values()),
            "total_deficit_mbit": sum(v["deficit_mbit"] for v in users.values()),
            "effective_user_time_fraction": (sum(v["effective_time_s"] for v in users.values()) / (len(users) * duration)
                                             if users and duration else None),
        }
        return {"scenario": scene, "through_time_s": float(cutoff), "through_revision": revision,
                "window_start_s": start, "window_s": window_s, "duration_s": duration,
                "completed_intervals": len(intervals), "users": users, "intervals": intervals, "metrics": metrics,
                "semantics": "Closed prior post-control modeled saturated/backlogged rate integrals only; Mbps*s=Mbit. No measured traffic claim."}

    def planning_history(self, scene, horizon_s=None):
        """History retained in the projected end-of-next-interval window."""
        horizon = self.horizon_s if horizon_s is None else horizon_s
        if isinstance(horizon, bool) or not math.isfinite(float(horizon)) or horizon <= 0:
            raise ValueError("Planning horizon must be positive")
        if self.history_window_s is None:
            history = self.traffic_history(scene)
        else:
            retained = self.history_window_s - horizon
            if retained < 0:
                raise ValueError("Planning horizon exceeds fairness window")
            # window_s must be positive in the public observation API. A W=h
            # projected window intentionally retains no earlier interval.
            history = self.traffic_history(scene, window_s=retained if retained else None)
            if not retained:
                history["duration_s"], history["completed_intervals"], history["intervals"] = 0.0, 0, []
                history["window_start_s"] = history["through_time_s"]
                for value in history["users"].values():
                    for key in ("traffic_mbit", "served_mbit", "effective_served_mbit", "demand_mbit", "deficit_mbit",
                                "effective_time_s", "observed_duration_s"):
                        value[key] = 0.0
                    for key in ("mean_rate_mbps", "normalized_mean_rate", "effective_time_fraction"):
                        value[key] = None
                history["metrics"] = {"jain_cumulative_rate_all_admitted": None,
                                      "jain_cumulative_demand_normalized_all_admitted": None,
                                      "total_traffic_mbit": 0.0, "total_deficit_mbit": 0.0,
                                      "effective_user_time_fraction": None}
            history["window_s"] = retained
        history["planning_projection"] = {"horizon_s": float(horizon), "target_window_s": self.history_window_s,
                                           "projected_end_time_s": history["through_time_s"] + horizon,
                                           "future_rates_observed": False}
        return history

    def fairness_status(self, scene, *, through_time_s=None, through_revision=None, window_s=None):
        from .fairness import score_candidate
        state = self.current_state(scene)
        cutoff = state["time_s"] if through_time_s is None else through_time_s
        if cutoff != state["time_s"] or (through_revision is not None and through_revision != state["revision"]):
            raise ValueError("fairness_status assesses current state; use traffic_history for historical cutoffs")
        history = self.traffic_history(scene, through_time_s=cutoff, through_revision=through_revision,
                                       window_s=self.history_window_s if window_s is None else window_s)
        score = score_candidate(state["allocations"], {uid: u["cqi"] for uid, u in state["users"].items()},
                                state["contracts"], self.planning_history(scene), horizon_s=self.horizon_s,
                                reference_allocations=state["allocations"])
        return {"scenario": scene, "revision": state["revision"], "time_s": state["time_s"],
                "history": history, "current_score": score, "horizon_s": self.horizon_s,
                "optimization_history_window_s": self.history_window_s,
                "query_changes_optimization_window": False,
                "objective_priority": ["maximize_effective_users", "maximize_historical_log_utility",
                                       "maximize_effective_throughput_mbps", "minimize_bandwidth_change"]}

    def read_context(self, scene, frame_index=None, phase="post_control"):
        context = super().read_context(scene, frame_index, phase)
        state = context["state"]
        context["traffic_history"] = self.traffic_history(
            scene, through_time_s=state["time_s"], through_revision=state["revision"], window_s=self.history_window_s)
        context["objective_contract"] = ["maximize effective users in the fixed admitted cohort",
                                         "then maximize log utility of historical normalized average throughput",
                                         "then maximize effective throughput; minimize bandwidth changes on ties"]
        return context

    @staticmethod
    def _controller_provenance(decision):
        """Do not infer an actual model invocation from an agent's tool name."""
        executed = decision.get("agent_executed")
        if executed is not None and not isinstance(executed, bool):
            raise ValueError("agent_executed provenance must be boolean when supplied")
        controller = decision.get("controller") or ("agent_tool" if decision.get("agent") else "unspecified_controller")
        if not isinstance(controller, str) or not controller.strip():
            raise ValueError("Controller provenance must be a nonempty string")
        if executed is True:
            source = "agent_control_commit"
        elif executed is False:
            source = "numeric_validation_control_commit" if controller == "numeric_validation" else "non_agent_control_commit"
        else:
            source = "agent_tool_control_commit_execution_unverified" if decision.get("agent") else "validated_controller_commit"
        return {"controller": controller, "agent_executed": executed, "allocation_source": source,
                "provenance_scope": "Declared caller execution mode; tool naming alone does not prove an API invocation."}

    def _materialize_control(self, before, after, events, decision, changed):
        """Keep old transaction implementation, label controller origins honestly."""
        provenance = self._controller_provenance(decision)
        decision["execution_provenance"] = provenance
        for event in events:
            if event.get("reason") == "agent_submitted_validated_plan":
                event["reason"] = "controller_submitted_validated_plan"
        super()._materialize_control(before, after, events, decision, changed)
        scene, revision = after["scenario"], after["revision"]
        identifiers = [(f"{scene}:state_event:{revision:04d}", False)]
        identifiers.extend((f"{scene}:allocation:{uid}:{revision}", True) for uid in changed)
        for identifier, allocation in identifiers:
            row = self.db.execute("SELECT properties_json FROM nodes WHERE id=?", (identifier,)).fetchone()
            properties = json.loads(row[0])
            properties.update(controller=provenance["controller"], agent_executed=provenance["agent_executed"],
                              execution_provenance=provenance)
            if allocation:
                properties["source"] = provenance["allocation_source"]
            self.db.execute("UPDATE nodes SET properties_json=? WHERE id=?", (packed(properties), identifier))

    def commit_allocations(self, scene, allocations, *, expected_revision, decision):
        """Commit only a maximum-service plan without regressing historical utility.

        The count-first guard uses the exact planner. At equal count historical
        log utility precedes effective throughput, permitting a throughput
        tradeoff that improves long-term fairness. History is recomputed inside
        the revision-bound transaction; submitted model claims are not trusted.
        """
        from .fairness import optimize_fair_service, score_candidate, compare_scores
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        if not isinstance(decision, dict) or not decision:
            raise ValueError("Nonempty decision provenance is required")
        decision = json.loads(packed(decision))  # Validate/copy before writing anything.
        decision["execution_provenance"] = self._controller_provenance(decision)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            before = self.current_state(scene)
            if before["revision"] != expected_revision:
                raise ValueError("Stale graph revision; re-read current state")
            if before["time_s"] == 0:
                raise ValueError("Static terminal import must not be reoptimized or controlled at t0")
            candidate = self._normalize_candidate(before, allocations)
            changed = [uid for uid in candidate if candidate[uid] != before["allocations"][uid]]
            if not changed:
                raise ValueError("No allocation changes; record a hold decision without a commit")
            evaluation = evaluate_service(list(before["users"].values()), candidate, before["contracts"])
            if any("rate_above_maximum" in u["violations"] for u in evaluation["users"].values()):
                raise ValueError("Candidate exceeds a service rate upper bound")
            old_metrics, new_metrics = before["metrics"], evaluation["metrics"]
            history = self.planning_history(scene)
            cqis = {uid: user["cqi"] for uid, user in before["users"].items()}
            best = optimize_fair_service(before["allocations"], cqis, before["contracts"],
                                         history, horizon_s=self.horizon_s)
            if new_metrics["effective_users"] < best["metrics"]["effective_users"]:
                raise ValueError("Candidate does not maximize effective-user count")
            old_score = score_candidate(before["allocations"], cqis, before["contracts"], history,
                                        horizon_s=self.horizon_s, reference_allocations=before["allocations"])
            new_score = score_candidate(candidate, cqis, before["contracts"], history,
                                        horizon_s=self.horizon_s, reference_allocations=before["allocations"])
            if not new_score["resource_valid"]:
                raise ValueError("Candidate violates hard resource constraints: " + "; ".join(new_score["violations"]))
            # Churn is only a planner tie breaker, not grounds to reject a
            # submitted resource-valid fairness-equivalent allocation.
            if compare_scores(new_score["objective"][:3] + [0.0], old_score["objective"][:3] + [0.0]) < 0:
                raise ValueError("Candidate regresses the effective-user/historical-fairness objective")
            decision["verified_fairness_objective"] = {
                "before": old_score["objective"], "after": new_score["objective"],
                "history_through_time_s": history["through_time_s"],
                "history_through_revision": history["through_revision"],
                "horizon_s": self.horizon_s, "window_s": self.history_window_s,
            }
            after = deepcopy(before)
            revision, frame, time_s = before["revision"] + 1, before["frame_index"], before["time_s"]
            after.update(revision=revision, phase="post_control", allocations=candidate,
                         metrics=new_metrics, allocation_policy=POLICY,
                         last_control={"revision": revision, "decision": decision, "changed_user_ids": changed})
            for uid in after["users"]:
                after["users"][uid]["service"] = evaluation["users"][uid]
                after["users"][uid]["admitted"] = evaluation["users"][uid]["admitted"]
            emitted = []

            def event(kind, uid, old, new, reason, episode_id=None):
                value = {"event_id": f"{scene}:control_event:{revision:04d}:{len(emitted):04d}",
                         "scenario": scene, "event_type": kind, "user_id": uid,
                         "time_s": time_s, "frame_index": frame, "revision": revision,
                         "before": old, "after": new, "reason": reason, "episode_id": episode_id,
                         "cause": "allocation_control", "decision": decision}
                emitted.append(value)
                return value["event_id"]

            open_episodes = {e["user_id"]: e for e in after["episodes"] if e["is_open"]}
            for uid in changed:
                event("ALLOCATION_CHANGED", uid, before["allocations"][uid], candidate[uid], "controller_submitted_validated_plan")
            for uid in candidate:
                old, new = before["users"][uid]["service"], after["users"][uid]["service"]
                if old["effective"] and not new["effective"]:
                    eid = f"{scene}:episode:{uid}:control:{revision:04d}"
                    onset = event("SERVICE_DEGRADED", uid, old, new, new["violations"], eid)
                    ep = {"episode_id": eid, "scenario": scene, "user_id": uid,
                          "start_time_s": time_s, "end_time_s": None, "last_observed_time_s": time_s,
                          "observed_duration_s": 0, "is_open": True, "reason": new["violations"],
                          "onset_event_id": onset, "recovery_event_id": None}
                    after["episodes"].append(ep)
                    open_episodes[uid] = ep
                elif not old["effective"] and new["effective"]:
                    if uid not in open_episodes:
                        raise RuntimeError(f"Missing degradation episode for recovered user {uid}")
                    ep = open_episodes.pop(uid)
                    recovery = event("SERVICE_RECOVERED", uid, old, new, "allocation_control_restored_service", ep["episode_id"])
                    ep.update(end_time_s=time_s, last_observed_time_s=time_s,
                              observed_duration_s=time_s - ep["start_time_s"], is_open=False,
                              recovery_event_id=recovery)
            after["active_episodes"] = {e["user_id"]: e for e in after["episodes"] if e["is_open"]}
            self._materialize_control(before, after, emitted, decision, changed)
            self.db.execute("INSERT INTO control_commits VALUES(?,?,?,?,?,?,?)",
                            (scene, revision, frame, time_s, packed(decision), packed(before), packed(after)))
            self.db.execute("UPDATE heads SET revision=? WHERE scenario=?", (revision, scene))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return after

