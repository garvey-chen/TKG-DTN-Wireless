"""Semantic checks; deterministic model doubles here are NOT experiment results."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wireless_exp import ROOT
from wireless_exp.data import load_users, scenario_path
from wireless_exp.llm import LoggedLLM
from wireless_exp.metrics import audit_state, empty_state, rate_mbps
from wireless_exp.policies import OriginalAgent, OriginalRule, load_legacy


class GraphTestLLM:
    """Exercise the original graph plumbing only, without imitating a real model."""

    failures = 0
    user_id = None

    def invoke(self, messages):
        text = messages[-1].content
        answer = "5" if "single integer number" in text else "I recommend using URLLC slice."
        return SimpleNamespace(content=answer)


@pytest.mark.parametrize("method", ["wireless_kb", "wireless_nkb"])
def test_original_graph_preserves_previous_users_and_resets_trials(method):
    agent = OriginalAgent(method, GraphTestLLM())
    observation = {
        "user_id": "1", "location": "(0, 0, 1.5)", "cqi": 13,
        "request": "I want to use remote surgery equipment",
    }
    result = agent.step(observation)
    assert not result["allocation_failed"]
    before = agent.snapshot()
    agent.step(observation | {"user_id": "2"})
    after = agent.snapshot()
    assert len(before["urllc_slice"]["users"]) == 1
    assert len(after["urllc_slice"]["users"]) == 2
    assert after["urllc_slice"]["resource_usage"] == 10
    audit = audit_state(after, {"1": "URLLC", "2": "URLLC"}, True)
    assert audit["valid_correct_users"] == 2
    assert audit["violations"] == []
    agent.module.reset_network_state()
    assert agent.snapshot()["urllc_slice"]["users"] == []


def test_snapshot_mutation_does_not_change_policy_state():
    module = load_legacy("prompt")
    snapshot = module.get_current_network_state()
    snapshot["embb_slice"]["users"].append({"user_id": "fake"})
    assert module.get_current_network_state()["embb_slice"]["users"] == []


def test_ground_truth_is_not_a_policy_input():
    user = load_users(scenario_path("center"), 1)[0]
    assert "ground_truth" not in user.observation()
    assert "Request_Label" not in user.observation()
    assert user.ground_truth == "URLLC"


def test_data_is_strict_and_never_falls_back(tmp_path):
    invalid = tmp_path / "bad.csv"
    invalid.write_text("RX_ID,CQI\n1,13\n")
    with pytest.raises(ValueError, match="Missing CSV"):
        load_users(invalid)
    with pytest.raises(ValueError, match="contains 30"):
        load_users(scenario_path("north"), 31)


def test_auditor_catches_model_rate_hallucination_without_repair():
    state = empty_state()
    state["embb_slice"]["users"] = [
        {"user_id": "1", "bandwidth": 6, "cqi": 1, "rate": 200, "latency": 50}
    ]
    state["embb_slice"]["resource_usage"] = 6
    state["total_users"] = 1
    result = audit_state(state, {"1": "eMBB"}, True)
    assert result["admitted_users"] == 1
    assert result["resource_feasible_users"] == 0
    assert result["reported_throughput_mbps"] == 200
    assert result["raw_calculated_throughput_mbps"] == pytest.approx(rate_mbps(6, 1))
    assert result["resource_feasible_throughput_mbps"] == 0
    assert "1:reported_rate_mismatch" in result["violations"]
    assert state["embb_slice"]["users"][0]["rate"] == 200


def test_original_rule_single_user_matches_formula():
    rule = OriginalRule()
    user = load_users(scenario_path("center"), 1)[0]
    rule.step_oracle(user)
    metrics = audit_state(rule.snapshot(), {user.user_id: user.ground_truth}, False)
    assert metrics["raw_calculated_throughput_mbps"] == pytest.approx(rate_mbps(5, 13))
    assert metrics["valid_correct_users"] == 1


def test_upstream_infeasible_fallback_is_exposed_not_renamed_optimal():
    import numpy as np
    rule = load_legacy("rule")
    bandwidths, rates, success = rule.solve_resource_allocation(
        90, 20, np.ones(20), 6, 20, 100, 400, 10
    )
    assert success  # A known upstream defect, deliberately preserved.
    assert any(b < 6 or rate < 100 for b, rate in zip(bandwidths, rates))


def settings():
    return {
        "api_key": "TEST_SECRET_DO_NOT_LOG", "base_url": "https://example.invalid/v1",
        "model": "test-model", "temperature": 0, "max_tokens": 16,
        "timeout_seconds": 1, "max_retries": 0, "max_calls_per_run": 2,
    }


def test_transport_records_usage_without_credential(tmp_path, monkeypatch):
    client = LoggedLLM(settings(), tmp_path / "calls.jsonl")
    monkeypatch.setattr(client.session, "post", lambda *a, **kw: SimpleNamespace(
        status_code=200,
        json=lambda: {"model": "test-model", "choices": [
            {"message": {"content": "response"}, "finish_reason": "stop"}
        ], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
    ))
    assert client.complete([{"role": "user", "content": "request"}]) == "response"
    assert client.prompt_tokens == 10
    log = client.log_path.read_text()
    assert "TEST_SECRET_DO_NOT_LOG" not in log
    assert json.loads(log)["response"] == "response"


def test_truncated_model_output_fails_closed(tmp_path, monkeypatch):
    client = LoggedLLM(settings(), tmp_path / "calls.jsonl")
    monkeypatch.setattr(client.session, "post", lambda *a, **kw: SimpleNamespace(
        status_code=200,
        json=lambda: {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
    ))
    with pytest.raises(RuntimeError, match="truncated"):
        client.complete([{"role": "user", "content": "request"}])
    assert client.failures == 1


def test_api_failure_cannot_be_counted_as_heuristic_success():
    class FailedLLM(GraphTestLLM):
        def invoke(self, messages):
            self.failures += 1
            raise RuntimeError("simulated transport failure")
    agent = OriginalAgent("wireless_kb", FailedLLM())
    with pytest.raises(RuntimeError, match="API transport"):
        agent.step({"user_id": "1", "location": "(0,0,0)", "request": "I want to watch 4K video", "cqi": 13})
    assert agent.snapshot()["embb_slice"]["users"] == []


def test_scene_generation_is_seeded_and_never_overwrites(tmp_path):
    from wireless_exp.scene import generate_scene
    first = generate_scene("center", 3, 42, tmp_path / "first")
    second = generate_scene("center", 3, 42, tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()
    assert len(load_users(first)) == 3
    with pytest.raises(ValueError, match="overwrite"):
        generate_scene("center", 3, 42, tmp_path / "first")


def test_report_excludes_failed_trials(tmp_path):
    from wireless_exp.report import report
    (tmp_path / "failed").mkdir()
    (tmp_path / "failed/summary.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="No complete"):
        report(tmp_path)
