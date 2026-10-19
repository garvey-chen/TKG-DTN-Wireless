"""Exercise the actual completion scheduler with synthetic latency, never APIs.

Each method is tested in its own interpreter to preserve archived import
isolation. Temporary outputs are deleted; these are not experiment results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace

HERE = Path(__file__).resolve().parent


def check_event_times():
    from run_completion_driven import event_times
    checks = [(0, 1, [1.0]), (1, 2.5, [2.0, 2.5]), (2.5, 3, [3.0]),
              (3, 3, []), (2.5, 2.75, [2.75]), (299.5, 300, [300.0]),
              (2, 2.0000001, [2.0000001]), (2.9999999, 3, [3.0])]
    for start, end, expected in checks:
        assert event_times(start, end) == expected, (start, end)
    for start, end in [(2, 1), (float("nan"), 2), (1, float("inf"))]:
        try:
            event_times(start, end)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid event range was accepted")
    return len(checks)


class NoModel:
    """The actual worker receives counters only; it cannot call a model."""
    session = SimpleNamespace(close=lambda: None)

    def statistics(self):
        return {key: 0 for key in ("llm_calls", "api_attempts", "llm_failures", "prompt_tokens", "completion_tokens")}

    def complete(self, *args, **kwargs):
        raise AssertionError("An integration test attempted a model call")


def test_worker(method):
    import run_completion_driven as worker
    runtime, provenance = worker.load_runtime(method)
    import continuation.fairness_graph as selected_graph
    assert runtime.resolve() in Path(selected_graph.__file__).resolve().parents, "Archived runtime import escaped"
    duration_sequence = [1.5, 0.5, 7.0, 289.0, 2.0]
    expected_sources = [1.0, 2.5, 3.0, 10.0, 299.0]
    first_source_allocation = None
    checkpoint_10 = None
    with tempfile.TemporaryDirectory(prefix=f"scheduler_test_{method}_") as temporary:
        output = Path(temporary) / "output"
        output.mkdir()
        args = SimpleNamespace(method=method, scene="center", output=output,
                               input_dir=worker.DEFAULT_INPUT)
        experiment = worker.Experiment(args, runtime, provenance, settings=None)
        experiment.llm = NoModel()

        def synthetic_infer(self):
            nonlocal first_source_allocation, checkpoint_10
            index = len(self.decisions)
            assert index < len(duration_sequence), "Unexpected extra decision launch"
            source = float(self.state["time_s"])
            assert source == expected_sources[index], "Next round did not begin at the prior completion"
            latest = {row["RX_ID"]: row for row in self.samples[math.floor(source)]}
            for uid, user in self.state["users"].items():
                assert user["cqi"] == int(latest[uid]["CQI"])
                assert user["location"] == [float(latest[uid][key]) for key in ("X", "Y", "Z")]
            if self.graph:
                history = self.graph.traffic_history(self.scene)
                assert history["through_time_s"] == source
                assert all(item["end_time_s"] <= source for item in history["intervals"])
            if source == 10:
                assert all(math.isclose(item["observed_duration_s"], 10) for item in self.ledger.values())
                checkpoint_10 = {"all_30_users_integrated_to_10s": True,
                                 "last_sample_time_s": math.floor(source),
                                 "completed_decisions": len(self.decisions)}
            allocation = deepcopy(self.state["allocations"])
            status = "held"
            if index == 0:
                first_source_allocation = deepcopy(allocation)
                if method == "tkg_dtn":
                    from continuation.fairness import optimize_fair_service
                    before = deepcopy(self.state)
                    original_history = deepcopy(self.graph.traffic_history(self.scene))
                    candidate = self.graph.fork_candidate(self.scratch_path / "test_only_candidate.sqlite")
                    try:
                        state = candidate.current_state(self.scene)
                        plan = optimize_fair_service(state["allocations"],
                            {uid: user["cqi"] for uid, user in state["users"].items()}, state["contracts"],
                            candidate.planning_history(self.scene), horizon_s=10.0)
                        allocation = deepcopy(plan["allocations"])
                        assert allocation != before["allocations"]
                        candidate.commit_allocations(self.scene, allocation,
                            expected_revision=state["revision"], decision={"controller": "numeric_validation",
                                "agent_executed": False, "decision_id": "test_source_candidate"})
                        assert self.graph.current_state(self.scene) == before
                        assert self.graph.traffic_history(self.scene) == original_history
                    finally:
                        candidate.close()
                    status = "reallocated"
                else:
                    # Test the actual WirelessAgent executor's hard guard with
                    # a deliberately invalid, fully synthetic proposal.
                    uid = next(iter(allocation))
                    allocation[uid]["bandwidth"] = 10000
                    status = "approved"
            elif index == 2:
                status = "failed"
            elif index == 4:
                status = "reallocated" if method == "tkg_dtn" else "approved"
                allocation = deepcopy(first_source_allocation)
            duration = duration_sequence[index]
            return {"decision_id": f"synthetic_{index}", "method": method, "scenario": self.scene,
                "decision_index": index + 1, "source_time_s": source,
                "source_sample_time_s": math.floor(source), "source_revision": self.state["revision"],
                "source_frame_index": self.state["frame_index"], "completed_time_s": source + duration,
                "scheduled_apply_time_s": source + duration, "decision_wall_seconds": duration,
                "policy_reported_seconds": duration, "policy_status": status,
                "policy_result": {"status": status, "synthetic_test": True},
                "allocations": allocation, "allocation_before": deepcopy(self.state["allocations"]),
                "applied_time_s": None, "checked_time_s": None,
                "llm_calls": 0, "api_attempts": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0}

        experiment.infer = MethodType(synthetic_infer, experiment)
        complete = False
        try:
            with redirect_stdout(io.StringIO()):
                experiment.run()
            complete = True
            assert len(experiment.decisions) == 5
            assert experiment.state["time_s"] == 300
            assert set(experiment.metrics) == set(range(301))
            assert experiment.decisions[-1]["status"] == "terminal_drop"
            assert experiment.decisions[-1]["completed_time_s"] == 301
            assert experiment.decisions[-1]["applied_time_s"] is None
            assert experiment.decisions[-1]["allocation_after"] == experiment.decisions[-1]["allocation_before"]
            assert experiment.decisions[1]["status"] == "held"
            assert experiment.decisions[2]["status"] == "policy_failed"
            if method == "tkg_dtn":
                assert experiment.decisions[0]["status"] == "reallocated"
                assert experiment.decisions[0]["applied_time_s"] == 2.5
                assert experiment.graph.current_state("center")["frame_index"] == 301
            else:
                assert experiment.decisions[0]["status"] == "rejected_stale_or_invalid"
            histories = defaultdict(list)
            for interval in experiment.intervals:
                histories[interval["user_id"]].append(interval)
                assert interval["telemetry_time_s"] == math.floor(interval["start_time_s"])
            assert set(histories) == {str(uid) for uid in range(1, 31)}
            for uid, intervals in histories.items():
                assert intervals[0]["start_time_s"] == 0 and intervals[-1]["end_time_s"] == 300
                assert all(a["end_time_s"] == b["start_time_s"] for a, b in zip(intervals, intervals[1:]))
                assert math.isclose(sum(row["duration_s"] for row in intervals), 300, abs_tol=1e-10)
                assert math.isclose(sum(row["traffic_mbit"] for row in intervals), experiment.ledger[uid]["traffic_mbit"], abs_tol=1e-8)
                assert any(row["start_time_s"] == 2 and row["end_time_s"] == 2.5 for row in intervals)
                assert any(row["start_time_s"] == 2.5 and row["end_time_s"] == 3 for row in intervals)
                if uid not in experiment.state["allocations"]:
                    assert experiment.ledger[uid]["traffic_mbit"] == 0
                    assert experiment.ledger[uid]["effective_time_s"] == 0
                    assert experiment.ledger[uid]["observed_duration_s"] == 300
            result = {"method": method, "passed": True, "api_calls": 0,
                "experiment_result": False, "runtime": str(runtime), "duration_s": 300,
                "checkpoint_10s": checkpoint_10, "decision_sources_s": expected_sources,
                "completion_times_s": [2.5, 3, 10, 299, 301],
                "status_counts": dict(Counter(row["status"] for row in experiment.decisions)),
                "all_30_user_duration_s": 300, "interval_rows": len(experiment.intervals),
                "candidate_does_not_affect_service_before_completion": True,
                "full_301_metric_samples": True, "terminal_proposal_dropped": True}
        finally:
            experiment.finalize(complete)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("tkg_dtn", "wireless_agent"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(test_worker(args.worker), ensure_ascii=False))
        return
    boundary_cases = check_event_times()
    results = []
    for method in ("tkg_dtn", "wireless_agent"):
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", method],
                               check=True, text=True, capture_output=True)
        results.append(json.loads(child.stdout))
    print(json.dumps({"passed": True, "api_calls": 0, "event_boundary_cases": boundary_cases,
                      "workers": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
