"""Incremental-history invariants, exact resource optimality and reset safety."""

import itertools
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_tkg import graph, observation, allocated
from wireless_exp.metrics import SLICE_LIMITS, empty_state, rate_mbps
from wireless_exp.tkg.graph_store import normalized, replay_events
from wireless_exp.tkg.resources import legal_bandwidths, optimize_fixed_users, bandwidth_options
from wireless_exp.tkg.tools import WirelessTools
from wireless_exp.tkg.lifecycle import experiment_lock, reset_graph


@pytest.mark.parametrize("slice_type,cqis", [("eMBB", [5, 7, 9, 12, 15]), ("URLLC", [3, 5, 7, 9, 11, 13, 15])])
def test_fixed_user_optimizer_matches_independent_exhaustive_search(slice_type, cqis):
    state = empty_state()
    limits = SLICE_LIMITS[slice_type]
    users = [{"user_id": str(i), "cqi": q, "bandwidth": min(legal_bandwidths(slice_type, q)),
              "rate": rate_mbps(min(legal_bandwidths(slice_type, q)), q), "latency": limits["latency"][0]}
             for i, q in enumerate(cqis)]
    state[limits["key"]]["users"] = users
    state = normalized(state)
    saved = deepcopy(state)
    solution = optimize_fixed_users(state)
    assert state == saved
    assert solution["feasible"]
    brute = max(sum(rate_mbps(b, q) for b, q in zip(bands, cqis))
                for bands in itertools.product(*(legal_bandwidths(slice_type, q) for q in cqis)) if sum(bands) <= limits["capacity"])
    assert sum(u["rate"] for u in solution["candidate_state"][limits["key"]]["users"]) == pytest.approx(brute)


def test_exact_bandwidth_options_and_pinning():
    state = empty_state()
    cert = bandwidth_options(state, observation(cqi=7), "eMBB")
    assert cert["minimum_new_user_bandwidth_mhz"] == 13
    assert cert["feasible_after_adjustment_mhz"] == list(range(13, 21))
    assert not bandwidth_options(state, observation(cqi=1), "eMBB")["admission_feasible"]
    assert optimize_fixed_users(allocated(), pinned_bandwidth={"1": 8})["candidate_state"]["embb_slice"]["users"][0]["bandwidth"] == 8
    assert not optimize_fixed_users(allocated(), pinned_bandwidth={"1": 1})["feasible"]
    with pytest.raises(ValueError):
        legal_bandwidths("eMBB", 7.5)


def test_real_unchanged_allocation_survives_arrivals_rejections_and_unchanged_cqi(graph):
    graph.arrive(observation(), 1)
    graph.apply_decision(allocated(), {"user_id": "1", "status": "admitted"}, graph.revision)
    original_revision = graph.revision
    graph.arrive(observation("2"), 2)
    graph.apply_decision(graph.state, {"user_id": "2", "status": "rejected"}, graph.revision)
    graph.update_channel("1", 13, 3)
    rows = graph.connection.query("MATCH (a:WirelessAllocation {run_id:$rid}) RETURN a.valid_from AS start,a.valid_to AS end", {"rid": graph.run_id})
    assert rows == [{"start": original_revision, "end": None}]
    historical = graph.state_at_revision(original_revision)
    assert historical["state"] == normalized(allocated())
    assert set(historical["users"]) == {"1"}
    events = [json.loads(line) for line in (graph.output / "events.jsonl").read_text().splitlines()]
    assert all("state" not in event for event in events)
    assert events[-1]["delta"]["allocations_upsert"] == {}
    assert replay_events(events)["state"] == graph.state
    graph.depart("1", 4)
    assert graph.connection.query("MATCH (:WirelessUser {run_id:$rid,user_id:'1'})-[r:SERVED_BY|CURRENT_ALLOCATION]->() RETURN count(r) AS n", {"rid": graph.run_id}) == [{"n": 0}]
    graph.export()
    assert json.loads((graph.output / "mapping_stats.json").read_text())["allocation_nodes"] == 1


def test_real_exact_tools_no_nested_llm_and_separate_reallocation_commit(graph, tmp_path):
    class NoLLM:
        def invoke(self, *args, **kwargs):
            raise AssertionError("Resource tools must not invoke an LLM")
    graph.arrive(observation(), 1)
    graph.apply_decision(allocated(), {"user_id": "1", "status": "admitted"}, graph.revision)
    graph.arrive(observation("2", cqi=7), 2)
    tools = WirelessTools(graph, NoLLM(), tmp_path)
    tools.start("2")
    tools.context()
    with pytest.raises(ValueError, match="Rejection denied"):
        tools.reject_request("eMBB", "guess")
    before = graph.read_state()
    preview = tools.plan_reallocation()
    assert "candidate_state" not in preview
    assert graph.read_state() == before
    with pytest.raises(ValueError, match="Wrong commit"):
        tools.commit_allocation(preview["plan_id"])
    assert tools.commit_reallocation(preview["plan_id"])["status"] == "reallocated"
    assert tools.result is None
    plan = tools.plan_allocation("eMBB")
    assert plan["feasible"] and plan["allocation"]["bandwidth"] >= 13
    tools.commit_allocation(plan["plan_id"])
    assert graph.read_state()["total_users"] == 2


def test_reset_backup_failure_prevents_delete(tmp_path):
    calls = []
    class BrokenBackup:
        def query(self, query, parameters=None):
            calls.append(query)
            if "count(n)" in query:
                return [{"nodes": 2}]
            raise OSError("simulated failed graph read")
    with pytest.raises(OSError):
        reset_graph(BrokenBackup(), tmp_path / "backup")
    assert not any("DELETE" in query for query in calls)


def test_concurrent_matrix_lock_refuses_second_cleanup():
    with experiment_lock():
        with pytest.raises(RuntimeError, match="Another wireless matrix"):
            with experiment_lock():
                pytest.fail("second lock unexpectedly entered")
