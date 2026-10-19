"""Event timestamps around the unchanged, archived PDF fairness graph.

The worker must put the PDF r003 source runtime first on ``sys.path`` before
importing this module. Only the scheduler's graph/time representation changes;
planning and allocation guards remain the archived implementation.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3

from continuation.fairness_graph import FairnessWirelessGraph


TIMING_SEMANTICS = {
    "frame_index": "Monotone event index, not a telemetry sample index or elapsed seconds.",
    "telemetry": "Integer-second input samples; latest available sample is held between samples.",
    "candidate": "Each decision uses an isolated SQLite backup containing source-time state and completed history only.",
    "execution": "A completed candidate is revalidated on the real graph at its exact completion time; no rounding to a sample boundary.",
    "traffic": "Each telemetry/completion event closes the previous post-control rate integral. Same-time commits add no traffic.",
    "wall_time": "Measured complete decision duration advances simulated network time.",
}


class EventFairnessGraph(FairnessWirelessGraph):
    """Preserve original graph transactions while separating event/sample time."""

    def __init__(self, database, run_id=None, *, static_terminal_dir,
                 history_window_s=None, horizon_s=10.0):
        self.event_static_terminal_dir = Path(static_terminal_dir).resolve()
        super().__init__(database, run_id, static_terminal_dir=static_terminal_dir,
                         history_window_s=history_window_s, horizon_s=horizon_s)
        self.db.executescript("""
CREATE TABLE IF NOT EXISTS event_timing(
 scenario TEXT NOT NULL, event_index INTEGER NOT NULL,
 event_time_s REAL NOT NULL, telemetry_sample_time_s REAL NOT NULL,
 event_kind TEXT NOT NULL, source_rows_json TEXT NOT NULL,
 PRIMARY KEY(scenario,event_index), UNIQUE(scenario,event_time_s));
""")
        semantics = json.dumps(TIMING_SEMANTICS, ensure_ascii=False, sort_keys=True)
        self.db.execute("INSERT INTO metadata VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                        ("event_time_semantics", semantics))
        # Existing control metadata is scoped to this new real/candidate graph.
        # Archived input databases and source files are never opened for writes.
        self.db.execute("INSERT INTO metadata VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                        ("control_semantics", semantics))

    @staticmethod
    def _telemetry_rows(rows, scene, sample_time_s):
        observations = []
        for original in rows:
            row = deepcopy(original)
            if "Scenario" in row and row["Scenario"] != scene:
                raise ValueError("Mixed scenes in telemetry input")
            if "Time_s" in row and float(row["Time_s"]) != sample_time_s:
                raise ValueError("Telemetry timestamp differs from sample_time_s")
            # Original sync_frame validates these as graph frame/time. Preserve
            # sample provenance in event_timing instead of relabeling the input.
            for key in ("Scenario", "Frame_Index", "Time_s"):
                row.pop(key, None)
            observations.append(row)
        return observations

    def event_metadata(self, scene, event_index=None):
        if event_index is None:
            event_index = self.current_state(scene)["frame_index"]
        row = self.db.execute(
            "SELECT event_index,event_time_s,telemetry_sample_time_s,event_kind FROM event_timing WHERE scenario=? AND event_index=?",
            (scene, event_index)).fetchone()
        if row is None:
            raise KeyError(f"Missing event timing: {scene}/{event_index}")
        value = dict(row)
        value["telemetry_age_s"] = value["event_time_s"] - value["telemetry_sample_time_s"]
        return value

    def _materialize(self, state, previous, events):
        pending = getattr(self, "_event_sync_pending", None)
        if pending is None:
            raise ValueError("Use sync_event so telemetry provenance accompanies every graph event")
        super()._materialize(state, previous, events)
        # The archived sync_frame invokes this hook inside its transaction.
        # Timing provenance and the original graph/traffic writes therefore
        # succeed together or roll back together.
        self.db.execute("INSERT INTO event_timing VALUES(?,?,?,?,?,?)",
            (state["scenario"], state["frame_index"], state["time_s"],
             pending["sample_time_s"], pending["event_kind"], pending["source_rows_json"]))

    def sync_event(self, scene, time_s, rows, sample_time_s, scene_info=None):
        """Close the previous real interval and synchronize one monotone event.

        Integer telemetry events and fractional decision-completion events both
        receive graph event indices. Repeating an identical timestamp/sample is
        a read-only operation, including after an allocation commit at that time.
        """
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Event time must be a finite nonnegative number")
        if (isinstance(sample_time_s, bool) or not isinstance(sample_time_s, (int, float))
                or not math.isfinite(sample_time_s) or sample_time_s < 0
                or not float(sample_time_s).is_integer()):
            raise ValueError("Telemetry sample time must be a nonnegative integer second")
        if sample_time_s > time_s:
            raise ValueError("Future telemetry is forbidden")
        if math.floor(time_s) != sample_time_s:
            raise ValueError("Use the latest available integer-second telemetry sample")
        normalized = self._telemetry_rows(list(rows), scene, sample_time_s)
        serialized = json.dumps(normalized, sort_keys=True, ensure_ascii=False, allow_nan=False)
        head = self.db.execute("SELECT * FROM heads WHERE scenario=?", (scene,)).fetchone()
        if head is None:
            if time_s != 0 or sample_time_s != 0:
                raise ValueError("Initialize the inherited static state at t=0")
            event_index, revision = 0, 0
        else:
            if time_s < head["time_s"]:
                raise ValueError("Event time cannot move backwards")
            if time_s == head["time_s"]:
                timing = self.db.execute(
                    "SELECT * FROM event_timing WHERE scenario=? AND event_index=?",
                    (scene, head["frame_index"])).fetchone()
                if (timing is None or timing["telemetry_sample_time_s"] != sample_time_s
                        or timing["source_rows_json"] != serialized):
                    raise ValueError("A same-time event cannot replace its telemetry")
                return self.current_state(scene)
            previous = self.event_metadata(scene)
            # The ledger must observe every intervening channel sample. The
            # scheduler may split an interval, but cannot skip an integer tick.
            if sample_time_s > previous["telemetry_sample_time_s"] + 1:
                raise ValueError("Synchronize every intervening integer telemetry sample")
            if sample_time_s > previous["telemetry_sample_time_s"] and time_s != sample_time_s:
                raise ValueError("Synchronize the intervening integer telemetry sample at its own timestamp first")
            if sample_time_s < previous["telemetry_sample_time_s"]:
                raise ValueError("Telemetry sample time cannot move backwards")
            event_index, revision = head["frame_index"] + 1, head["revision"]
        event_kind = "telemetry_sample" if time_s == sample_time_s else "decision_completion"
        self._event_sync_pending = {"sample_time_s": float(sample_time_s),
                                   "event_kind": event_kind, "source_rows_json": serialized}
        try:
            return super().sync_frame(scene, event_index, float(time_s), normalized,
                expected_revision=revision, target_users=30, scene_info=scene_info,
                initialization_mode="static_terminal")
        finally:
            self._event_sync_pending = None

    def read_context(self, scene, frame_index=None, phase="post_control"):
        context = super().read_context(scene, frame_index=frame_index, phase=phase)
        context["event_timing"] = self.event_metadata(scene, context["state"]["frame_index"])
        context["control_time_semantics"] = (
            "This decision reads a frozen source-time snapshot. Its explicit allocation result is a candidate; "
            "the scheduler revalidates it on the real graph at measured completion time. "
            "Graph frame_index is a monotone event index. Telemetry sample time is recorded separately. "
            "Completed service-history intervals are inherited exactly; a same-time commit creates no traffic.")
        return context

    def fork_candidate(self, path):
        """Copy the complete committed graph/history without sharing mutable state."""
        destination = Path(path)
        if destination.resolve() == self.database.resolve() or destination.exists():
            raise FileExistsError(f"Candidate database must be a new path: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(destination) as target:
            self.db.backup(target)
        candidate = type(self)(destination, self.run_id,
            static_terminal_dir=self.event_static_terminal_dir,
            history_window_s=self.history_window_s, horizon_s=self.horizon_s)
        candidate.db.execute("INSERT INTO metadata VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            ("candidate_snapshot", json.dumps({
                "source_database": str(self.database.resolve()),
                "source_heads": [dict(row) for row in self.db.execute("SELECT * FROM heads ORDER BY scenario")],
                "scope": "isolated source-time planning sandbox; its commits are not real execution",
            }, ensure_ascii=False, sort_keys=True)))
        return candidate


EventGraphAdapter = EventFairnessGraph
