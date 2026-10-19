"""Revision-bound control extension for the existing temporal wireless graph.

Telemetry ``frames`` remain immutable pre-control snapshots. Successful controls
append immutable before/after records at the same simulated timestamp. The
post-control state is held until the next telemetry frame; model wall time is
not simulated network time. The original monitoring graph and schema are not
modified, so earlier baseline source hashes remain reproducible.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math

from .graph import TemporalWirelessGraph, packed
from .service import SLICE_LIMITS, RESOURCE_TOLERANCE, evaluate_service


POLICY = "revision_bound_agent_control; fixed_cohort_and_slices"


class ControlledWirelessGraph(TemporalWirelessGraph):
    """Add causal allocation transactions without overwriting frame evidence."""

    def __init__(self, database, run_id=None):
        super().__init__(database, run_id)
        self.db.executescript("""
CREATE TABLE IF NOT EXISTS control_commits(
 scenario TEXT NOT NULL, revision INTEGER NOT NULL, frame_index INTEGER NOT NULL,
 time_s REAL NOT NULL, decision_json TEXT NOT NULL,
 before_state_json TEXT NOT NULL, after_state_json TEXT NOT NULL,
 PRIMARY KEY(scenario,revision));
CREATE INDEX IF NOT EXISTS controls_by_frame ON control_commits(scenario,frame_index,revision);
""")
        if self.db.execute("SELECT 1 FROM metadata WHERE key='control_semantics'").fetchone() is None:
            self.db.execute("INSERT INTO metadata VALUES(?,?)", ("control_semantics", packed({
                            "frame_snapshot": "Immutable pre-control telemetry snapshot.",
                            "control_snapshot": "Immutable before/after state at the same simulation timestamp.",
                            "interval": "Post-control state held on [time_s,next_frame_time_s).",
                            "revision": "Advances for telemetry and successful allocation commits.",
                            "wall_time": "Processing latency is recorded separately; no simulated delay is claimed.",
            })))

    def current_state(self, scene):
        head = self.db.execute("SELECT frame_index FROM heads WHERE scenario=?", (scene,)).fetchone()
        if head is None:
            raise KeyError(f"No synchronized frame for {scene}")
        return self.state_at(scene, head["frame_index"])

    def state_at(self, scene, frame_index, phase="post_control"):
        if phase not in ("pre_control", "post_control"):
            raise ValueError("phase must be pre_control or post_control")
        if phase == "post_control":
            row = self.db.execute(
                "SELECT after_state_json FROM control_commits WHERE scenario=? AND frame_index=? ORDER BY revision DESC LIMIT 1",
                (scene, frame_index),
            ).fetchone()
            if row is not None:
                return json.loads(row[0])
        return super().state_at(scene, frame_index)

    def history(self, scene, through_time_s=None, user_id=None, event_type=None, through_revision=None):
        query, values = "SELECT properties_json FROM events WHERE scenario=?", [scene]
        for column, value in (("time_s", through_time_s), ("revision", through_revision)):
            if value is not None:
                query += f" AND {column}<=?"
                values.append(value)
        if user_id is not None:
            query += " AND user_id=?"
            values.append(str(user_id))
        if event_type is not None:
            query += " AND event_type=?"
            values.append(event_type)
        query += " ORDER BY revision,event_id"
        return [json.loads(row[0]) for row in self.db.execute(query, values)]

    def read_context(self, scene, frame_index=None, phase="post_control"):
        if phase not in ("pre_control", "post_control"):
            raise ValueError("phase must be pre_control or post_control")
        if frame_index is None:
            current = self.current_state(scene)
            state = current if phase == "post_control" else self.state_at(scene, current["frame_index"], phase)
        else:
            state = self.state_at(scene, frame_index, phase)
        return {
            "run_id": self.run_id, "backend": "sqlite_local_property_graph", "state": state,
            "events": self.history(scene, through_time_s=state["time_s"], through_revision=state["revision"]),
            "constraints": deepcopy(SLICE_LIMITS),
            "objective_contract": ["maximize effective users within the fixed admitted cohort",
                                   "then maximize effective throughput; report fairness alongside throughput"],
            "assessment_scope": "Rate, integer bandwidth, assigned slice and pool capacity; latency is unmeasured.",
            "control_time_semantics": "Synchronous control after observing this frame; post-control state is held until the next frame.",
        }

    def _active_allocation_id(self, scene, uid):
        rows = self.db.execute(
            "SELECT target FROM edges WHERE source=? AND relation='CURRENT_ALLOCATION'",
            (f"{scene}:user:{uid}",),
        ).fetchall()
        if len(rows) != 1:
            raise RuntimeError(f"Expected exactly one current allocation for {scene}/{uid}")
        return rows[0][0]

    def _materialize(self, state, previous, events):
        # Called inside the base class's frame transaction before state insertion.
        state["allocation_policy"] = POLICY
        state["phase"] = "pre_control"
        super()._materialize(state, previous, events)
        scene, frame = state["scenario"], state["frame_index"]
        for uid in state["users"]:
            status_id = f"{scene}:service_status:{uid}:{frame:04d}"
            props = json.loads(self.db.execute("SELECT properties_json FROM nodes WHERE id=?", (status_id,)).fetchone()[0])
            props["phase"] = "pre_control"
            self.db.execute("UPDATE nodes SET properties_json=? WHERE id=?", (packed(props), status_id))
            if uid in state["allocations"]:
                self.db.execute("DELETE FROM edges WHERE source=? AND relation='EVALUATES_ALLOCATION'", (status_id,))
                self._edge("EVALUATES_ALLOCATION", status_id, self._active_allocation_id(scene, uid), scene)
        # Explicit phase is also visible in graph exports, not just snapshots.
        fid = f"{scene}:frame:{frame:04d}"
        row = self.db.execute("SELECT properties_json FROM nodes WHERE id=?", (fid,)).fetchone()
        props = json.loads(row[0])
        props["phase"] = "pre_control"
        self.db.execute("UPDATE nodes SET properties_json=? WHERE id=?", (packed(props), fid))

    @staticmethod
    def _normalize_candidate(state, allocations):
        if not isinstance(allocations, dict):
            raise ValueError("allocations must be a complete user-keyed dictionary")
        candidate = {}
        for key, allocation in allocations.items():
            uid = str(key)
            if uid in candidate or not isinstance(allocation, dict):
                raise ValueError("Duplicate user or malformed allocation")
            if str(allocation.get("user_id", uid)) != uid:
                raise ValueError("Allocation user_id mismatch")
            name = allocation.get("slice_type")
            if uid not in state["allocations"] or name != state["allocations"][uid]["slice_type"]:
                raise ValueError("Control must retain the fixed admitted cohort and assigned slices")
            bandwidth = allocation.get("bandwidth")
            if isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, float)):
                raise ValueError("Bandwidth must be a finite integer number of MHz")
            if not math.isfinite(bandwidth) or not float(bandwidth).is_integer():
                raise ValueError("Bandwidth must be a finite integer number of MHz")
            limits = SLICE_LIMITS[name]
            if not limits["bandwidth"][0] <= bandwidth <= limits["bandwidth"][1]:
                raise ValueError(f"Bandwidth outside per-user bounds for {uid}")
            candidate[uid] = {"user_id": uid, "slice_type": name, "bandwidth": int(bandwidth)}
        if set(candidate) != set(state["allocations"]):
            raise ValueError("Control must retain the complete fixed admitted cohort")
        for name, limits in SLICE_LIMITS.items():
            usage = sum(a["bandwidth"] for a in candidate.values() if a["slice_type"] == name)
            if usage > limits["capacity"] + RESOURCE_TOLERANCE:
                raise ValueError(f"Slice capacity exceeded: {name}")
        return candidate

    def commit_allocations(self, scene, allocations, *, expected_revision, decision):
        """Atomically apply a fixed-cohort plan based on the current revision.

        Bandwidth, pool, slice and rate upper bounds are hard constraints. A
        physically unattainable minimum rate may remain unmet. The guard forbids
        reducing effective-user count, or effective throughput at equal count;
        it does not claim that the submitted plan is globally optimal.
        """
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        if not isinstance(decision, dict) or not decision:
            raise ValueError("Nonempty decision provenance is required")
        decision = json.loads(packed(decision))  # Validate/copy before writing anything.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            before = self.current_state(scene)
            if before["revision"] != expected_revision:
                raise ValueError("Stale graph revision; re-read current state")
            candidate = self._normalize_candidate(before, allocations)
            changed = [uid for uid in candidate if candidate[uid] != before["allocations"][uid]]
            if not changed:
                raise ValueError("No allocation changes; record a hold decision without a commit")
            evaluation = evaluate_service(list(before["users"].values()), candidate, before["contracts"])
            if any("rate_above_maximum" in u["violations"] for u in evaluation["users"].values()):
                raise ValueError("Candidate exceeds a service rate upper bound")
            old_metrics, new_metrics = before["metrics"], evaluation["metrics"]
            if new_metrics["effective_users"] < old_metrics["effective_users"] or (
                new_metrics["effective_users"] == old_metrics["effective_users"]
                and new_metrics["effective_throughput_mbps"] < old_metrics["effective_throughput_mbps"] - 1e-8
            ):
                raise ValueError("Candidate regresses the effective-user/throughput objective")
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
                event("ALLOCATION_CHANGED", uid, before["allocations"][uid], candidate[uid], "agent_submitted_validated_plan")
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

    def _materialize_control(self, before, after, events, decision, changed):
        scene, frame = after["scenario"], after["frame_index"]
        revision, time_s = after["revision"], after["time_s"]
        state_event, fid = f"{scene}:state_event:{revision:04d}", f"{scene}:frame:{frame:04d}"
        self._node("StateEvent", state_event, scene, event_type="CONTROL_COMMIT", frame_index=frame,
                   time_s=time_s, logical_time=time_s, revision=revision, phase="post_control",
                   observed_at=datetime.now(timezone.utc).isoformat(), changed_user_ids=changed,
                   users_upsert={uid: after["users"][uid] for uid in changed}, allocations_changed=True,
                   decision=decision, metrics_before=before["metrics"], metrics_after=after["metrics"],
                   state_sha256=hashlib.sha256(packed(after).encode()).hexdigest())
        self._edge("FOLLOWS", state_event, f"{scene}:state_event:{before['revision']:04d}", scene)
        for uid in changed:
            old_id = self._active_allocation_id(scene, uid)
            old_props = json.loads(self.db.execute("SELECT properties_json FROM nodes WHERE id=?", (old_id,)).fetchone()[0])
            old_props.update(valid_to=revision, valid_to_time_s=time_s)
            self.db.execute("UPDATE nodes SET properties_json=? WHERE id=?", (packed(old_props), old_id))
            aid, user_node = f"{scene}:allocation:{uid}:{revision}", f"{scene}:user:{uid}"
            self._node("Allocation", aid, scene, **after["allocations"][uid], valid_from=revision, valid_to=None,
                       valid_from_time_s=time_s, valid_to_time_s=None,
                       source="agent_control_commit", decision=decision, control_state_event_id=state_event)
            self._edge("HAS_ALLOCATION", user_node, aid, scene)
            self.db.execute("DELETE FROM edges WHERE source=? AND relation='CURRENT_ALLOCATION'", (user_node,))
            self._edge("CURRENT_ALLOCATION", user_node, aid, scene)
            self._edge("ON_SLICE", aid, f"{scene}:slice:{after['allocations'][uid]['slice_type']}", scene)
        # Every user has a post-control assessment; telemetry evidence is shared,
        # not duplicated or retroactively changed. Historical status edges keep
        # pointing to the allocation version that was actually evaluated.
        for uid, user in after["users"].items():
            oid = f"{scene}:observation:{uid}:{frame:04d}"
            status_id = f"{scene}:service_status:{uid}:{frame:04d}:control:{revision}"
            user_id = f"{scene}:user:{uid}"
            user_props = json.loads(self.db.execute("SELECT properties_json FROM nodes WHERE id=?", (user_id,)).fetchone()[0])
            user_props.update(revision=revision, phase="post_control")
            self.db.execute("UPDATE nodes SET properties_json=? WHERE id=?", (packed(user_props), user_id))
            self._node("ServiceStatus", status_id, scene, frame_index=frame, time_s=time_s,
                       revision=revision, phase="post_control", control_state_event_id=state_event, **user["service"])
            self._edge("HAS_SERVICE_STATUS", oid, status_id, scene)
            if uid in after["allocations"]:
                self._edge("EVALUATES_ALLOCATION", status_id, self._active_allocation_id(scene, uid), scene)
        for episode in after["episodes"]:
            self._node("ServiceEpisode", episode["episode_id"], scene, **{k: v for k, v in episode.items() if k != "scenario"})
            self._edge("HAS_EPISODE", f"{scene}:user:{episode['user_id']}", episode["episode_id"], scene)
        for value in events:
            self.db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                            tuple(value[k] for k in ("event_id", "scenario", "frame_index", "time_s", "revision", "event_type", "user_id")) + (packed(value),))
            self._node("ServiceEvent", value["event_id"], scene, **{k: v for k, v in value.items() if k != "scenario"})
            self._edge("HAS_EVENT", fid, value["event_id"], scene)
            self._edge("AFFECTS_USER", value["event_id"], f"{scene}:user:{value['user_id']}", scene)
            self._edge("EVIDENCED_BY", value["event_id"], f"{scene}:observation:{value['user_id']}:{frame:04d}", scene)
            if value["episode_id"]:
                self._edge("IN_EPISODE", value["event_id"], value["episode_id"], scene)
