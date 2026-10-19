"""Transactional local temporal property graph for frame-by-frame service assurance.

SQLite persists actual nodes/edges, immutable frame snapshots, observations and
events. It is an explicit offline backend; no Neo4j or agent execution is implied.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import uuid

from .service import (SLICE_LIMITS, INITIALIZATION_MODES, initialize_cohort,
                      evaluate_service, optimize_allocations)
from .schema import NODE_TYPES, RELATION_TYPES, definition


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class TemporalWirelessGraph:
    def __init__(self, database: Path, run_id: str | None = None):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.database, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY, kind TEXT NOT NULL, scenario TEXT NOT NULL, properties_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS edges(id TEXT PRIMARY KEY, relation TEXT NOT NULL, scenario TEXT NOT NULL,
 source TEXT NOT NULL REFERENCES nodes(id), target TEXT NOT NULL REFERENCES nodes(id), properties_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS heads(scenario TEXT PRIMARY KEY, revision INTEGER NOT NULL, frame_index INTEGER NOT NULL, time_s REAL NOT NULL);
CREATE TABLE IF NOT EXISTS frames(scenario TEXT NOT NULL, frame_index INTEGER NOT NULL, time_s REAL NOT NULL,
 revision INTEGER NOT NULL, state_json TEXT NOT NULL, PRIMARY KEY(scenario,frame_index), UNIQUE(scenario,revision));
CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY, scenario TEXT NOT NULL, frame_index INTEGER NOT NULL,
 time_s REAL NOT NULL, revision INTEGER NOT NULL, event_type TEXT NOT NULL, user_id TEXT, properties_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_by_time ON events(scenario,time_s,user_id);
CREATE INDEX IF NOT EXISTS nodes_by_kind ON nodes(scenario,kind);
CREATE INDEX IF NOT EXISTS edges_by_source ON edges(source,relation);
""")
        row = self.db.execute("SELECT value_json FROM metadata WHERE key='run_id'").fetchone()
        if row:
            self.run_id = json.loads(row[0])
            if run_id is not None and run_id != self.run_id:
                self.db.close()
                raise ValueError("Database belongs to a different run_id")
        else:
            self.run_id = run_id or "wireless_dynamic_" + uuid.uuid4().hex
            self.db.execute("BEGIN IMMEDIATE")
            try:
                for key, value in {"run_id": self.run_id, "backend": "sqlite_local_property_graph",
                                   "schema": definition(), "format_version": 1}.items():
                    self.db.execute("INSERT INTO metadata VALUES(?,?)", (key, packed(value)))
                for kind, description in NODE_TYPES.items():
                    self._node("SchemaType", "schema:" + kind, "", name=kind, description=description)
                for relation, endpoints in RELATION_TYPES.items():
                    if relation in ("INSTANCE_OF", "SCHEMA_RELATION") or any(k not in NODE_TYPES for k in endpoints):
                        continue
                    self._edge("SCHEMA_RELATION", "schema:" + endpoints[0], "schema:" + endpoints[1], "",
                               key_suffix=relation, relation_name=relation)
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                self.db.close()
                raise

    def close(self):
        self.db.close()

    def _node(self, kind, identifier, scenario, **props):
        if kind not in NODE_TYPES:
            raise ValueError(f"Unknown graph node kind: {kind}")
        props = {**props, "run_id": self.run_id, "scenario": scenario}
        self.db.execute("INSERT INTO nodes VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET properties_json=excluded.properties_json",
                        (identifier, kind, scenario, packed(props)))
        if kind != "SchemaType":
            self._edge("INSTANCE_OF", identifier, "schema:" + kind, scenario)

    def _edge(self, relation, source, target, scenario, key_suffix="", **props):
        if relation not in RELATION_TYPES and relation not in ("INSTANCE_OF", "SCHEMA_RELATION"):
            raise ValueError(f"Unknown graph relationship: {relation}")
        identifier = packed([source, relation, target, key_suffix])
        self.db.execute("INSERT INTO edges VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET properties_json=excluded.properties_json",
                        (identifier, relation, scenario, source, target, packed({**props, "run_id": self.run_id})))

    def current_state(self, scene):
        row = self.db.execute("SELECT state_json FROM frames WHERE scenario=? ORDER BY frame_index DESC LIMIT 1", (scene,)).fetchone()
        if row is None:
            raise KeyError(f"No synchronized frame for {scene}")
        return json.loads(row[0])

    def state_at(self, scene, frame_index):
        row = self.db.execute("SELECT state_json FROM frames WHERE scenario=? AND frame_index=?", (scene, frame_index)).fetchone()
        if row is None:
            raise KeyError(f"Frame not synchronized: {scene}/{frame_index}")
        return json.loads(row[0])

    def history(self, scene, through_time_s=None, user_id=None, event_type=None):
        query, values = "SELECT properties_json FROM events WHERE scenario=?", [scene]
        if through_time_s is not None:
            query += " AND time_s<=?"
            values.append(through_time_s)
        if user_id is not None:
            query += " AND user_id=?"
            values.append(str(user_id))
        if event_type is not None:
            query += " AND event_type=?"
            values.append(event_type)
        query += " ORDER BY frame_index,event_id"
        return [json.loads(row[0]) for row in self.db.execute(query, values)]

    def read_context(self, scene, frame_index=None):
        state = self.current_state(scene) if frame_index is None else self.state_at(scene, frame_index)
        preview = optimize_allocations(state["allocations"], {uid: u["cqi"] for uid, u in state["users"].items()})
        return {
            "run_id": self.run_id, "backend": "sqlite_local_property_graph", "state": state,
            "events": self.history(scene, through_time_s=state["time_s"]),
            "constraints": deepcopy(SLICE_LIMITS), "reallocation_preview": preview,
            "objective_contract": ["maximize effective users within the fixed admitted cohort",
                                   "then maximize effective throughput; report fairness alongside throughput"],
            "preview_scope": "Read-only fixed-cohort throughput optimization; no evictions or fairness optimization. No action has been applied.",
            "assessment_scope": "Rate, integer bandwidth, assigned slice and pool capacity; latency is unmeasured.",
        }

    @staticmethod
    def _observations(rows, scene, frame_index, time_s):
        users = {}
        for row in rows:
            uid = str(row["RX_ID"])
            if not uid.strip() or uid in users:
                raise ValueError("Missing or duplicate RX_ID within a frame")
            cqi = float(row["CQI"])
            location = [float(row[k]) for k in ("X", "Y", "Z")]
            snr, power, los = float(row["SNR_dB"]), float(row["RX_Power_dBm"]), float(row["LOS"])
            if not all(math.isfinite(v) for v in [cqi, *location, snr, power, los]):
                raise ValueError("Non-finite telemetry")
            if not cqi.is_integer() or not 1 <= cqi <= 15 or los not in (0, 1):
                raise ValueError("Invalid CQI/LOS telemetry")
            if not str(row["User_Request"]).strip():
                raise ValueError("Empty request")
            if "Scenario" in row and row["Scenario"] != scene:
                raise ValueError("Mixed scenarios in a frame")
            if "Frame_Index" in row and float(row["Frame_Index"]) != frame_index:
                raise ValueError("Mixed frame indices")
            if "Time_s" in row and float(row["Time_s"]) != time_s:
                raise ValueError("Mixed telemetry timestamps")
            users[uid] = {"user_id": uid, "location": location, "cqi": int(cqi), "snr_db": snr,
                          "rx_power_dbm": power, "los": bool(los), "request": row["User_Request"]}
        if not users:
            raise ValueError("Empty telemetry frame")
        return users

    def sync_frame(self, scene, frame_index, time_s, rows, *, expected_revision, target_users=10,
                   scene_info=None, rate_drop_fraction=0.1,
                   initialization_mode="feasible_admission"):
        """Commit the complete frame, including physically degraded states, atomically.

        Environmental updates never change allocations. Stale/duplicate/incomplete
        frames are rejected without partially updating any node, edge or event.
        """
        if not re.fullmatch(r"[a-z][a-z0-9_]*", scene):
            raise ValueError("Invalid scenario identifier")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        if not math.isfinite(float(time_s)) or time_s < 0:
            raise ValueError("Invalid logical time")
        if not math.isfinite(rate_drop_fraction) or not 0 < rate_drop_fraction <= 1:
            raise ValueError("Rate drop fraction must be in (0,1]")
        if initialization_mode not in INITIALIZATION_MODES:
            raise ValueError(f"Unknown initialization_mode: {initialization_mode}")
        rows = list(rows)
        users = self._observations(rows, scene, frame_index, time_s)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            head = self.db.execute("SELECT * FROM heads WHERE scenario=?", (scene,)).fetchone()
            actual_revision = head["revision"] if head else 0
            if expected_revision != actual_revision:
                raise ValueError("Stale graph revision; re-read current state")
            previous = self.current_state(scene) if head else None
            if head:
                if initialization_mode != previous.get("initialization_mode", "feasible_admission"):
                    raise ValueError("Keep initialization_mode fixed within a run")
                if frame_index != head["frame_index"] + 1 or time_s <= head["time_s"]:
                    raise ValueError("Duplicate, skipped or out-of-order frame/time")
                if set(users) != set(previous["users"]):
                    raise ValueError("Incomplete or changed fixed user population")
                for row in rows:
                    uid = str(row["RX_ID"])
                    if users[uid]["request"] != previous["users"][uid]["request"]:
                        raise ValueError("A user's request changed inside the fixed-service experiment")
                    if row.get("Request_Label", previous["contracts"][uid]["slice_type"]) != previous["contracts"][uid]["slice_type"]:
                        raise ValueError("Pre-established service contract changed")
                allocations, contracts = previous["allocations"], previous["contracts"]
                admission_log = previous["admission_log"]
                episodes = deepcopy(previous["episodes"])
                initialization_evidence = previous.get("initialization_evidence")
                scene_info = previous["scene_info"]
                if rate_drop_fraction != previous["rate_drop_fraction"]:
                    raise ValueError("Keep the event detector threshold fixed within a run")
            else:
                if frame_index != 0 or time_s != 0:
                    raise ValueError("Initial frame must be frame 0 at t=0")
                initial = initialize_cohort(rows, target_users, initialization_mode=initialization_mode)
                allocations, contracts = initial["allocations"], initial["contracts"]
                admission_log, episodes = initial["admission_log"], []
                initialization_evidence = initial.get("initialization_evidence")
                scene_info = deepcopy(scene_info or {})
            evaluation = evaluate_service(rows, allocations, contracts)
            for uid in users:
                users[uid]["admitted"] = evaluation["users"][uid]["admitted"]
                users[uid]["service"] = evaluation["users"][uid]
            revision = actual_revision + 1
            emitted = []

            def event(event_type, uid, before=None, after=None, reason=None, episode_id=None):
                value = {"event_id": f"{scene}:event:{frame_index:04d}:{len(emitted):04d}",
                         "scenario": scene, "event_type": event_type, "user_id": uid,
                         "time_s": time_s, "frame_index": frame_index, "revision": revision,
                         "before": before, "after": after, "reason": reason, "episode_id": episode_id}
                emitted.append(value)
                return value["event_id"]

            open_episodes = {e["user_id"]: e for e in episodes if e["is_open"]}

            def begin_episode(uid, old_service, new_service):
                eid = f"{scene}:episode:{uid}:{frame_index:04d}"
                onset = event("SERVICE_DEGRADED", uid, old_service, new_service,
                              reason=new_service["violations"], episode_id=eid)
                episode = {"episode_id": eid, "scenario": scene, "user_id": uid,
                           "start_time_s": time_s, "end_time_s": None, "last_observed_time_s": time_s,
                           "observed_duration_s": 0, "is_open": True, "reason": new_service["violations"],
                           "onset_event_id": onset, "recovery_event_id": None}
                episodes.append(episode)
                open_episodes[uid] = episode

            for uid, current in users.items():
                if previous is None:
                    if current["admitted"]:
                        event("USER_ADMITTED", uid, after=allocations[uid], reason="known_service_contract_at_t0")
                        if current["service"]["degraded"]:
                            # Initial service failure is observed at t=0; no
                            # earlier effective state or history is invented.
                            begin_episode(uid, None, current["service"])
                    continue
                old = previous["users"][uid]
                if current["location"] != old["location"]:
                    event("POSITION_CHANGED", uid, old["location"], current["location"])
                if current["cqi"] != old["cqi"]:
                    event("CQI_CHANGED", uid, old["cqi"], current["cqi"],
                          reason="decreased" if current["cqi"] < old["cqi"] else "increased")
                if not current["admitted"]:
                    continue
                old_service, new_service = old["service"], current["service"]
                if old_service["rate_mbps"] > 0 and new_service["rate_mbps"] <= old_service["rate_mbps"] * (1 - rate_drop_fraction):
                    event("RATE_DROP", uid, old_service["rate_mbps"], new_service["rate_mbps"],
                          reason={"relative_drop_threshold": rate_drop_fraction,
                                  "sla_failed": not new_service["effective"]})
                if old_service["effective"] and not new_service["effective"]:
                    begin_episode(uid, old_service, new_service)
                elif not old_service["effective"] and new_service["effective"]:
                    episode = open_episodes.pop(uid)
                    recovery = event("SERVICE_RECOVERED", uid, old_service, new_service,
                                     reason="modeled_service_constraints_satisfied", episode_id=episode["episode_id"])
                    episode.update(end_time_s=time_s, last_observed_time_s=time_s,
                                   observed_duration_s=time_s - episode["start_time_s"], is_open=False,
                                   recovery_event_id=recovery)
            for episode in open_episodes.values():
                episode["last_observed_time_s"] = time_s
                episode["observed_duration_s"] = time_s - episode["start_time_s"]
            state = {"scenario": scene, "revision": revision, "frame_index": frame_index, "time_s": time_s,
                     "users": users, "allocations": deepcopy(allocations), "contracts": deepcopy(contracts),
                     "metrics": evaluation["metrics"], "admission_log": admission_log,
                     "episodes": episodes, "active_episodes": {e["user_id"]: e for e in episodes if e["is_open"]},
                     "scene_info": scene_info, "rate_drop_fraction": rate_drop_fraction,
                     "initialization_mode": initialization_mode,
                     "allocation_policy": ("frozen_t0_count_first_fair_allocation; monitoring_only"
                                           if initialization_mode == "fixed_population" else
                                           "frozen_t0_exact_throughput_allocation; monitoring_only")}
            if initialization_evidence is not None:
                state["initialization_evidence"] = deepcopy(initialization_evidence)
            self._materialize(state, previous, emitted)
            self.db.execute("INSERT INTO frames VALUES(?,?,?,?,?)", (scene, frame_index, time_s, revision, packed(state)))
            self.db.execute("INSERT INTO heads VALUES(?,?,?,?) ON CONFLICT(scenario) DO UPDATE SET revision=excluded.revision,frame_index=excluded.frame_index,time_s=excluded.time_s",
                            (scene, revision, frame_index, time_s))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        if self.current_state(scene) != state:
            raise RuntimeError("Persisted graph snapshot differs from synchronized frame")
        return state

    def _materialize(self, state, previous, events):
        scene, frame = state["scenario"], state["frame_index"]
        time, revision = state["time_s"], state["revision"]
        sid, fid, bs = f"{scene}:scenario", f"{scene}:frame:{frame:04d}", f"{scene}:bs:1"
        window = f"{scene}:minute:{int(time // 60):04d}"
        state_event = f"{scene}:state_event:{revision:04d}"
        if previous is None:
            self._node("Scenario", sid, scene, name=scene, contract_source="known_csv_services_at_t0")
            self._node("BaseStation", bs, scene, name=scene + " base station", **state["scene_info"])
            self._edge("HAS_BASE_STATION", sid, bs, scene)
            for name, limits in SLICE_LIMITS.items():
                self._node("Slice", f"{scene}:slice:{name}", scene, name=name, **limits)
                self._edge("PROVIDES", bs, f"{scene}:slice:{name}", scene)
        self._node("TimeWindow", window, scene, start_time_s=int(time // 60) * 60, end_time_s=(int(time // 60) + 1) * 60)
        self._edge("HAS_WINDOW", sid, window, scene)
        self._node("Frame", fid, scene, frame_index=frame, time_s=time, logical_time=time, revision=revision, metrics=state["metrics"])
        self._edge("CONTAINS_FRAME", window, fid, scene)
        if frame:
            self._edge("NEXT_FRAME", f"{scene}:frame:{frame-1:04d}", fid, scene)
        changed = {uid: user for uid, user in state["users"].items()
                   if previous is None or user != previous["users"][uid]}
        self._node("StateEvent", state_event, scene, event_type="FRAME_SYNC", frame_index=frame, time_s=time,
                   logical_time=time, revision=revision, observed_at=datetime.now(timezone.utc).isoformat(),
                   changed_user_ids=list(changed), users_upsert=changed,
                   allocations_changed=previous is None, state_sha256=hashlib.sha256(packed(state).encode()).hexdigest())
        if previous:
            self._edge("FOLLOWS", state_event, f"{scene}:state_event:{revision-1:04d}", scene)
        for uid, user in state["users"].items():
            user_node, oid = f"{scene}:user:{uid}", f"{scene}:observation:{uid}:{frame:04d}"
            status_id, req = f"{scene}:service_status:{uid}:{frame:04d}", f"{scene}:requirement:{uid}"
            self._node("User", user_node, scene, user_id=uid, request=user["request"], location=user["location"],
                       cqi=user["cqi"], admitted=user["admitted"], present=True, frame_index=frame, time_s=time, revision=revision)
            if previous is None:
                self._edge("SERVED_BY", user_node, bs, scene, radio_association=True, admitted=user["admitted"])
                self._node("ServiceRequirement", req, scene, user_id=uid, source="pre_established_service_contract", **state["contracts"][uid])
                self._edge("HAS_REQUIREMENT", user_node, req, scene)
                self._edge("REQUIRES_SLICE", req, f"{scene}:slice:{state['contracts'][uid]['slice_type']}", scene)
                if user["admitted"]:
                    aid = f"{scene}:allocation:{uid}:1"
                    self._node("Allocation", aid, scene, **state["allocations"][uid], valid_from=1, valid_to=None,
                               valid_from_time_s=0, valid_to_time_s=None, source="t0_exact_numeric_initialization")
                    self._edge("HAS_ALLOCATION", user_node, aid, scene)
                    self._edge("CURRENT_ALLOCATION", user_node, aid, scene)
                    self._edge("ON_SLICE", aid, f"{scene}:slice:{state['allocations'][uid]['slice_type']}", scene)
            self._node("Observation", oid, scene, user_id=uid, frame_index=frame, time_s=time, revision=revision,
                       location=user["location"], cqi=user["cqi"], snr_db=user["snr_db"],
                       rx_power_dbm=user["rx_power_dbm"], los=user["los"])
            self._edge("HAS_OBSERVATION", user_node, oid, scene)
            self._edge("IN_FRAME", oid, fid, scene)
            self._edge("RECORDED_IN", oid, state_event, scene)
            if frame:
                self._edge("NEXT_OBSERVATION", f"{scene}:observation:{uid}:{frame-1:04d}", oid, scene)
            self._node("ServiceStatus", status_id, scene, frame_index=frame, time_s=time, revision=revision, **user["service"])
            self._edge("HAS_SERVICE_STATUS", oid, status_id, scene)
            if user["admitted"]:
                self._edge("EVALUATES_ALLOCATION", status_id, f"{scene}:allocation:{uid}:1", scene)
        # Persist episodes before referencing them from the new service events.
        for episode in state["episodes"]:
            self._node("ServiceEpisode", episode["episode_id"], scene, **{k: v for k, v in episode.items() if k != "scenario"})
            self._edge("HAS_EPISODE", f"{scene}:user:{episode['user_id']}", episode["episode_id"], scene)
        for event in events:
            self.db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                            tuple(event[key] for key in ("event_id", "scenario", "frame_index", "time_s", "revision", "event_type", "user_id")) + (packed(event),))
            self._node("ServiceEvent", event["event_id"], scene, **{k: v for k, v in event.items() if k != "scenario"})
            self._edge("HAS_EVENT", fid, event["event_id"], scene)
            self._edge("AFFECTS_USER", event["event_id"], f"{scene}:user:{event['user_id']}", scene)
            self._edge("EVIDENCED_BY", event["event_id"], f"{scene}:observation:{event['user_id']}:{frame:04d}", scene)
            if event["episode_id"]:
                self._edge("IN_EPISODE", event["event_id"], event["episode_id"], scene)
