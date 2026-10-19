"""Targeted integration checks; temporary SQLite only, no API/model invocation."""

from copy import deepcopy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

from .fairness_graph import FairnessWirelessGraph
from .fairness import optimize_fair_service, score_candidate, compare_scores
from .fair_agent import FairWirelessTools
from .planner import validate_allocations
from .service import SLICE_LIMITS, evaluate_service


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/mobility_5min_static_terminal_20261014"
HANDOFF = ROOT / "data/static_terminal_handoff_20261014"
SCENES = ("center", "north", "south")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ContinuationChecks(unittest.TestCase):
    def test_import_control_and_all_closed_intervals(self):
        timeline = read_csv(INPUT / "timeline.csv")
        with tempfile.TemporaryDirectory(prefix="static_terminal_check_") as temporary:
            graphs = {name: FairnessWirelessGraph(Path(temporary) / f"{name}.sqlite",
                                                   static_terminal_dir=HANDOFF)
                      for name in ("frozen", "numeric")}
            terminal = {scene: json.loads((HANDOFF / scene / "terminal_step.json").read_text())
                        for scene in SCENES}
            starts, integrated, previous, observed_wrong_changes = {}, {}, {}, set()
            for timing in timeline:
                index, timestamp = int(timing["Frame_Index"]), float(timing["Time_s"])
                for scene in SCENES:
                    rows = read_csv(INPUT / timing[f"{scene}_csv"])
                    for method, graph in graphs.items():
                        key = (method, scene)
                        prior = previous.get(key)
                        state = graph.sync_frame(
                            scene, index, timestamp, rows,
                            expected_revision=prior["revision"] if prior else 0,
                            initialization_mode="static_terminal", target_users=30)
                        self.assertEqual(len(state["users"]), 30)
                        if index == 0:
                            starts[key] = deepcopy(state)
                            integrated[key] = {uid: {"traffic": 0.0, "effective": 0.0}
                                               for uid in state["allocations"]}
                            source = terminal[scene]
                            self.assertEqual(state["initialization_evidence"]["raw_static_state"], source["state"])
                            self.assertEqual(len(state["allocations"]), source["admitted_users"])
                            for allocation in source["allocations"]:
                                uid = str(allocation["user_id"])
                                self.assertEqual(state["allocations"][uid],
                                                 {"user_id": uid, "slice_type": allocation["slice"],
                                                  "bandwidth": allocation["bandwidth"]})
                                self.assertEqual(state["users"][uid]["legacy_latency_ms"], allocation["latency"])
                                self.assertAlmostEqual(state["users"][uid]["service"]["rate_mbps"], allocation["rate"], places=10)
                            with self.assertRaisesRegex(ValueError, "t0"):
                                graph.commit_allocations(scene, deepcopy(state["allocations"]),
                                                         expected_revision=state["revision"],
                                                         decision={"controller": "test", "agent_executed": False})
                        else:
                            for uid in prior["allocations"]:
                                service = prior["users"][uid]["service"]
                                integrated[key][uid]["traffic"] += service["rate_mbps"] * 10
                                integrated[key][uid]["effective"] += int(service["effective"]) * 10
                        if method == "numeric" and 0 < index < 30:
                            cqis = {uid: u["cqi"] for uid, u in state["users"].items()}
                            plan = optimize_fair_service(state["allocations"], cqis, state["contracts"],
                                                         graph.planning_history(scene))
                            self.assertTrue(plan["resource_valid"], plan["violations"])
                            independent = evaluate_service(rows, plan["allocations"], state["contracts"])
                            self.assertEqual(plan["objective"][0], independent["metrics"]["effective_users"])
                            self.assertEqual(plan["metrics"].keys(), independent["metrics"].keys())
                            for metric, value in plan["metrics"].items():
                                # The planner sorts user IDs while this independent
                                # evaluation retains CSV order. Floating sums may
                                # differ in the last bits; counts/flags stay exact.
                                if isinstance(value, float):
                                    self.assertAlmostEqual(value, independent["metrics"][metric],
                                                           delta=1e-10, msg=metric)
                                else:
                                    self.assertEqual(value, independent["metrics"][metric], metric)
                            if plan["allocations"] != state["allocations"]:
                                after = graph.commit_allocations(
                                    scene, plan["allocations"], expected_revision=state["revision"],
                                    decision={"controller": "numeric_validation", "agent_executed": False})
                                for uid in plan["inherited_wrong_slice_user_ids"]:
                                    if after["allocations"][uid] != state["allocations"][uid]:
                                        observed_wrong_changes.add((scene, uid))
                            else:
                                after = state
                            if scene == "south" and index == 1:
                                tools = FairWirelessTools(graph, Path(temporary) / "tools")
                                tools.start(scene, {"type": "validation"})
                                tools.get_decision_context()
                                options = tools.get_bandwidth_options("7")
                                self.assertEqual(options["qos_feasible_bandwidths_mhz"], [])
                                self.assertTrue(options["hard_resource_valid_bandwidths_mhz"])
                                migrated = deepcopy(after["allocations"])
                                migrated["7"]["slice_type"] = "eMBB"
                                with self.assertRaisesRegex(ValueError, "assigned slices"):
                                    graph.commit_allocations(
                                        scene, migrated, expected_revision=after["revision"],
                                        decision={"controller": "test", "agent_executed": False})
                                self.assertEqual(graph.current_state(scene), after)
                        else:
                            after = state
                        initial = starts[key]
                        self.assertEqual(set(after["allocations"]), set(initial["allocations"]))
                        self.assertEqual(after["contracts"], initial["contracts"])
                        for uid, user in after["users"].items():
                            if uid not in initial["allocations"]:
                                self.assertFalse(user["admitted"])
                                self.assertEqual(user["service"]["rate_mbps"], 0.0)
                            elif initial["allocations"][uid]["slice_type"] != initial["contracts"][uid]["slice_type"]:
                                self.assertFalse(user["service"]["effective"])
                                self.assertIn("wrong_slice", user["service"]["violations"])
                            if uid in initial["allocations"]:
                                self.assertEqual(after["allocations"][uid]["slice_type"],
                                                 initial["allocations"][uid]["slice_type"])
                        if method == "frozen":
                            self.assertEqual(after["allocations"], initial["allocations"])
                        previous[key] = after
                    if index == 0:
                        for field in ("allocations", "contracts", "users", "metrics"):
                            self.assertEqual(starts[("frozen", scene)][field], starts[("numeric", scene)][field])
            self.assertTrue(observed_wrong_changes, "Inherited wrong-slice bandwidth must remain adjustable")
            for method, graph in graphs.items():
                for scene in SCENES:
                    history = graph.traffic_history(scene)
                    self.assertEqual(history["duration_s"], 300)
                    self.assertEqual(history["completed_intervals"], 30 * len(starts[(method, scene)]["allocations"]))
                    self.assertEqual(len(graph.history(scene, event_type="STATIC_TERMINAL_IMPORTED")), 30)
                    for uid, expected in integrated[(method, scene)].items():
                        self.assertAlmostEqual(history["users"][uid]["traffic_mbit"], expected["traffic"], places=7)
                        self.assertEqual(history["users"][uid]["effective_time_s"], expected["effective"])
                self.assertEqual(graph.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(graph.db.execute("PRAGMA foreign_key_check").fetchall(), [])
                graph.close()

    def test_wrong_slice_dp_matches_exhaustive_enumeration(self):
        contracts = {}
        for uid, name in (("a", "eMBB"), ("b", "URLLC"), ("c", "eMBB")):
            limits = SLICE_LIMITS[name]
            contracts[uid] = {"slice_type": name, "min_rate_mbps": limits["rate"][0],
                              "max_rate_mbps": limits["rate"][1]}
        # Include both mismatch directions, beyond those present in the actual input.
        original = {"a": {"user_id": "a", "slice_type": "URLLC", "bandwidth": 1},
                    "b": {"user_id": "b", "slice_type": "eMBB", "bandwidth": 6},
                    "c": {"user_id": "c", "slice_type": "eMBB", "bandwidth": 7}}
        cqis = {"a": 15, "b": 15, "c": 15}
        history = {"duration_s": 10, "users": {uid: {"traffic_mbit": volume}
                                             for uid, volume in (("a", 10), ("b", 1000), ("c", 1000))}}
        plan = optimize_fair_service(original, cqis, contracts, history)
        best = None
        for bandwidths in itertools.product(range(1, 6), range(6, 21), range(6, 21)):
            candidate = {uid: dict(original[uid], bandwidth=bw)
                         for uid, bw in zip(original, bandwidths)}
            if not validate_allocations(candidate, cqis, contracts, original)["resource_valid"]:
                continue
            score = score_candidate(candidate, cqis, contracts, history, reference_allocations=original)
            if best is None or compare_scores(score, best) > 0:
                best = score
        self.assertIsNotNone(best)
        self.assertEqual(compare_scores(plan["score"], best), 0)
        self.assertEqual(plan["objective"][0], 1)
        self.assertEqual(plan["inherited_wrong_slice_user_ids"], ["a", "b"])

    def test_original_sources_remain_unchanged(self):
        record = json.loads((ROOT / "continuation/source_origin.json").read_text())
        for entry in record["original_modules"].values():
            self.assertEqual(hashlib.sha256(Path(entry["source"]).read_bytes()).hexdigest(), entry["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
