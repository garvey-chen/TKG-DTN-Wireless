"""Execution optimizations must retain graph freshness and decision evidence."""

import json
import re
from copy import deepcopy

import pytest

from test_tkg import graph, observation
from wireless_exp import ROOT
from wireless_exp.tkg.graph_store import WirelessGraph
from wireless_exp.tkg.model_view import compact_schema
from wireless_exp.tkg.rag_system_agent import NetworkRAGAgent
from wireless_exp.tkg.tools import WirelessTools


def test_real_context_reuses_one_verified_snapshot_without_aliasing(graph):
    graph.arrive(observation(), 1)
    before = graph.connection.calls
    context = graph.context("1")
    assert graph.connection.calls - before == 1  # Combined server read.
    assert context["state"] == graph.read_state()
    assert context["knowledge"] == (ROOT / "data/knowledge/Intent_Understand.txt").read_text()
    before = graph.connection.calls
    reused = graph.context("1", reuse_verified=True)
    assert graph.connection.calls - before == 1  # Server revision still checked.
    assert reused == context
    reused["user"]["cqi"] = 1
    reused["constraints"]["eMBB"]["capacity"] = 999
    assert graph.context("1", reuse_verified=True) == context
    graph.update_channel("1", 7, 2)
    assert graph.context("1", reuse_verified=True)["user"]["cqi"] == 7
    graph.depart("1", 3)
    with pytest.raises(ValueError, match="not currently present"):
        graph.context("1", reuse_verified=True)


def test_real_cache_observes_external_revision_and_commit_rejects_race(graph, tmp_path):
    graph.arrive(observation(), 1)
    tools = WirelessTools(graph, None, tmp_path)
    tools.start("1")
    tools.context()
    plan = tools.plan_allocation("eMBB")
    external = WirelessGraph(graph.connection, tmp_path / "external", run_id=graph.run_id)
    external.revision, external.logical_time = graph.revision, graph.logical_time
    external.users, external.state = deepcopy(graph.users), deepcopy(graph.state)
    external.update_channel("1", 7, 2)
    refreshed = graph.context("1", reuse_verified=True)
    assert refreshed["revision"] == external.revision
    assert refreshed["user"]["cqi"] == 7
    # Local revision is unchanged: only the transactional server guard can
    # catch this race. A cached preview must never bypass that guard.
    with pytest.raises(RuntimeError, match="Stale graph revision"):
        tools.commit_allocation(plan["plan_id"])
    assert external.read_state()["total_users"] == 0


@pytest.mark.parametrize("cqi,status", [(13, "admitted"), (1, "rejected")])
def test_real_prefetch_and_compact_evidence_complete_decisions(graph, tmp_path, cqi, status):
    graph.arrive(observation(cqi=cqi), 1)
    tools = WirelessTools(graph, None, tmp_path)
    tools.start("1")
    knowledge = (ROOT / "data/knowledge/Intent_Understand.txt").read_text()
    calls = []

    class ScriptedModel:
        def complete(self, messages):
            prompt = messages[-1]["content"]
            calls.append(prompt)
            if len(calls) == 1:
                # No model call was spent requesting the required context;
                # every KB entry (including conflicts) reaches classification.
                assert json.dumps(knowledge, ensure_ascii=False)[1:-1] in prompt
                action = {"tool": "plan_allocation", "args": {"slice_type": "eMBB"}}
            else:
                assert len(calls) == 2
                assert json.dumps(knowledge, ensure_ascii=False)[1:-1] in prompt
                assert "minimum_new_user_bandwidth_mhz" in prompt
                plan_id = re.findall(r'"plan_id"\s*:\s*"([^"]+)"', prompt)[-1]
                assert plan_id in tools.plans
                action = ({"tool": "commit_allocation", "args": {"plan_id": plan_id}} if status == "admitted"
                          else {"tool": "reject_request", "args": {"slice_type": "eMBB", "reason": "No legal bandwidth at observed CQI"}})
            return "思考: Use exact tool evidence.\n行动: " + json.dumps(action)

    agent = NetworkRAGAgent(ScriptedModel(), tools)
    list(agent.ask_stream("Handle user 1"))
    assert tools.result["status"] == status
    records = [json.loads(line) for line in (tmp_path / "tool_calls.jsonl").read_text().splitlines()]
    assert records[0]["origin"] == "program_prefetch"
    assert records[0]["result"]["knowledge"] == knowledge
    plan = records[1]
    assert "rates_mbps" in plan["result"]["feasibility_certificate"]
    assert "rates_mbps" not in plan["model_observation"]["feasibility_certificate"]
    assert plan["result"]["feasible"] == (status == "admitted")
    assert graph.read_state() == graph.state
    graph.export()  # Checks full event replay, including post-write verification.


def test_real_full_evidence_remains_retrievable(graph, tmp_path):
    graph.arrive(observation(), 1)
    tools = WirelessTools(graph, None, tmp_path)
    tools.start("1")
    tools.context()
    full, message = tools.call("get_bandwidth_options", {"slice_type": "eMBB", "full_evidence": True})
    assert json.loads(message) == full and "rates_mbps" in full
    schema = graph.schema()
    compact = compact_schema(schema)
    assert set(compact) == {r["source_type"] for r in schema}
    assert sum(map(len, compact.values())) == sum(r["relation"] is not None for row in schema for r in row["relations"])
