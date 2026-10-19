"""Meaningful policy-guard and real Neo4j mapping checks.

Set WIRELESS_TEST_NEO4J_CONFIG to opt into integration tests. Each test creates
an isolated run; it never clears other graph data or uses an LLM credential.
"""

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from wireless_exp import ROOT
from wireless_exp.metrics import empty_state, rate_mbps
from wireless_exp.tkg.executor import resource_violations
from wireless_exp.tkg.graph_store import WirelessGraph, normalized
from wireless_exp.tkg.rag_system_agent import NetworkRAGAgent


def observation(uid="1", cqi=13):
    return {"user_id": uid, "cqi": cqi, "location": [1.0, 2.0], "request": "I want to watch 4K video"}


def allocated():
    state = empty_state()
    state["embb_slice"]["users"] = [{"user_id": "1", "bandwidth": 10, "cqi": 13,
                                      "rate": rate_mbps(10, 13), "latency": 50}]
    return normalized(state)


def test_common_guard_checks_resources_without_intent_labels():
    state = allocated()
    assert not resource_violations(state)
    bad = deepcopy(state)
    bad["embb_slice"]["users"][0]["bandwidth"] = 25
    assert "1:bandwidth" in resource_violations(normalized(bad))
    duplicate = deepcopy(state)
    duplicate["embb_slice"]["users"].append(deepcopy(duplicate["embb_slice"]["users"][0]))
    assert "1:duplicate_user" in resource_violations(normalized(duplicate))


def test_copied_parser_and_no_fabricated_terminal_result():
    agent = object.__new__(NetworkRAGAgent)
    thought, action = agent._parse_llm_action('思考：查图谱\n行动：{"tool":"get_decision_context","args":{}}')
    assert thought == "查图谱"
    assert action["tool"] == "get_decision_context"
    from types import SimpleNamespace
    agent.tools = SimpleNamespace(result=None)
    with pytest.raises(RuntimeError, match="without a committed"):
        agent._generate_final_answer("task", "")


def test_source_archive_provenance():
    manifest = json.loads((ROOT / "vendor/tkg_dtn/manifest.json").read_text())
    copy = ROOT / "vendor/tkg_dtn/rag_system_agent.py"
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == manifest["archived_sha256"]
    original = ROOT.parent / "rag_system_agent.py"
    if original.exists():
        assert hashlib.sha256(original.read_bytes()).hexdigest() == manifest["original_sha256"]


def test_guarded_baseline_rolls_back_invalid_whole_arrival(monkeypatch):
    from types import SimpleNamespace
    from wireless_exp.policies import OriginalAgent
    from wireless_exp.tkg.executor import GuardedWirelessAgent
    saved = {"state": allocated()}
    before = deepcopy(saved["state"])
    policy = object.__new__(GuardedWirelessAgent)
    policy.module = SimpleNamespace(
        get_current_network_state=lambda: saved["state"],
        update_network_state=lambda state: saved.update(state=state),
    )

    def invalid_workflow(self, incoming):
        saved["state"]["embb_slice"]["users"][0]["bandwidth"] = 25
        return {"slice_type": "eMBB"}

    monkeypatch.setattr(OriginalAgent, "step", invalid_workflow)
    result = policy.step(observation("2"))
    assert result["status"] == "rejected_by_common_guard"
    assert policy.snapshot() == before


@pytest.fixture
def graph(tmp_path):
    config = os.environ.get("WIRELESS_TEST_NEO4J_CONFIG")
    if not config:
        pytest.skip("Set WIRELESS_TEST_NEO4J_CONFIG for real Neo4j integration checks")
    from wireless_exp.tkg.connection import GraphConnection
    conn = GraphConnection(Path(config))
    try:
        twin = WirelessGraph(conn, tmp_path / "graph")
        twin.initialize()
        yield twin
    finally:
        conn.close()


def test_real_graph_mapping_history_and_dynamic_hooks(graph):
    assert len(graph.schema()) == 8
    graph.arrive(observation(), 1)
    first_revision = graph.revision
    assert graph.context("1")["user"]["cqi"] == 13
    with pytest.raises(ValueError, match="not currently present"):
        graph.context("future_user")
    graph.apply_decision(allocated(), {"user_id": "1", "status": "admitted"}, first_revision)
    assignment_revision = graph.revision
    with pytest.raises(ValueError, match="Stale"):
        graph.apply_decision(allocated(), {"status": "admitted"}, first_revision)
    graph.update_channel("1", 11, 2)
    assert graph.context("1")["user"]["cqi"] == 11
    assert graph.read_state()["embb_slice"]["users"][0]["rate"] == pytest.approx(rate_mbps(10, 11))
    historical = graph.connection.query(
        "MATCH (a:WirelessAllocation {run_id:$rid,valid_from:$rev}) RETURN a.cqi AS cqi,a.valid_to AS valid_to",
        {"rid": graph.run_id, "rev": assignment_revision},
    )
    assert historical == [{"cqi": 13, "valid_to": assignment_revision + 1}]
    before = graph.revision
    with pytest.raises(ValueError, match="Out-of-order"):
        graph.update_channel("1", 10, 1)
    assert graph.revision == before
    graph.depart("1", 3)
    assert graph.read_state()["total_users"] == 0
    assert {e["event_type"] for e in graph.history()} >= {"arrival", "decision", "channel_update", "departure"}
    graph.export()
    saved = json.loads((graph.output / "graph_snapshot.json").read_text())
    assert all("ground_truth" not in n for n in saved["nodes"])
    assert any(e["type"] == "SCHEMA_RELATION" for e in saved["edges"])


def test_real_graph_scope_and_optimistic_conflict(graph, tmp_path):
    other = WirelessGraph(graph.connection, tmp_path / "other")
    other.initialize()
    other.arrive(observation("other_user"), 1)
    assert not graph.search("other_user")
    stale = WirelessGraph(graph.connection, tmp_path / "stale", run_id=graph.run_id)
    stale.revision = graph.revision
    stale.logical_time = graph.logical_time
    graph.arrive(observation(), 1)
    with pytest.raises(RuntimeError, match="Stale graph revision"):
        stale.arrive(observation("stale_user"), 1)
    assert not graph.search("stale_user")
    with pytest.raises(ValueError, match="evaluator labels"):
        graph.arrive(observation("2") | {"ground_truth": "eMBB"}, 2)


def test_real_preview_is_nonmutating_and_commit_rejects_stale_or_invalid(graph, tmp_path):
    from types import SimpleNamespace
    from wireless_exp.tkg.tools import WirelessTools
    graph.arrive(observation(), 1)
    graph.apply_decision(allocated(), {"user_id": "1", "status": "admitted"}, graph.revision)
    graph.arrive(observation("2"), 2)
    tools = WirelessTools(graph, SimpleNamespace(failures=0), tmp_path)
    tools.start("2")
    tools.context()
    before = graph.read_state()
    invalid = tools.plan_allocation("eMBB", bandwidth=1)
    assert not invalid["feasible"]
    assert graph.read_state() == before
    with pytest.raises(ValueError, match="Infeasible"):
        tools.commit_allocation(invalid["plan_id"])
    valid = tools.plan_allocation("eMBB", bandwidth=10)
    assert valid["feasible"]
    assert graph.read_state() == before
    graph.update_channel("2", 12, 3)
    with pytest.raises(ValueError, match="Stale"):
        tools.commit_allocation(valid["plan_id"])
    fresh = tools.plan_allocation("eMBB", bandwidth=10)
    assert tools.commit_allocation(fresh["plan_id"])["status"] == "admitted"
    assert graph.read_state()["total_users"] == 2
