"""Shared allocation regressions; model doubles are not experiment results."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from wireless_exp.metrics import audit_state, empty_state, rate_mbps
from wireless_exp.policies import load_legacy


@pytest.fixture(params=["agent_kb", "agent_nkb"])
def agent(request):
    module = load_legacy(request.param)
    module.reset_network_state()
    def no_model(*args, **kwargs):
        raise AssertionError("Numerical resource tools must not ask a model for bandwidth")
    module.llm = SimpleNamespace(invoke=no_model)
    return module


def arrival_state():
    """Saved center/user-21 state: 90 MHz used, 74 MHz existing minima."""
    state = empty_state()
    state["embb_slice"]["users"] = [
        {"user_id": uid, "cqi": q, "bandwidth": b,
         "rate": round(rate_mbps(b, q), 2), "latency": 40}
        for uid, q, b in [("4", 7, 14), ("6", 10, 16), ("9", 15, 7),
                          ("11", 8, 17), ("15", 6, 16), ("18", 5, 20)]
    ]
    state["embb_slice"].update(resource_usage=90, utilization_rate="100%")
    state["total_users"] = 6
    return state


def request_bandwidth(agent, cqi=7):
    return agent.beamforming_tool.invoke({
        "user_id": "21", "slice_type": "eMBB", "cqi": cqi,
        "request": "I want to use augmented reality navigation",
    })


def commit(agent, result, cqi=7):
    return agent.slice_allocation.invoke({
        "user_id": "21", "slice_type": "eMBB", "cqi": cqi,
        "bandwidth": result["allocated_bandwidth"], "rate": result["allocated_rate"],
        "latency": result["allocated_latency"], "resource_plan": result["resource_plan"],
    })


def test_joint_allocation_admits_and_assigns_surplus_by_marginal_rate(agent):
    agent.update_network_state(arrival_state())
    before = agent.get_current_network_state()
    result = request_bandwidth(agent)
    assert result["allocated_bandwidth"] == 13
    assert result["rate_valid"] and result["admission_feasible"]
    assert agent.get_current_network_state() == before
    commit(agent, result)
    after = agent.get_current_network_state()
    # All minima total 87 MHz. The remaining 3 go to CQI-15 user 9.
    assert {u["user_id"]: u["bandwidth"] for u in after["embb_slice"]["users"]} == {
        "4": 13, "6": 10, "9": 10, "11": 12, "15": 15, "18": 17, "21": 13,
    }
    labels = {u["user_id"]: "eMBB" for u in after["embb_slice"]["users"]}
    assert audit_state(after, labels, True)["violations"] == []


def test_unused_capacity_is_distributed_without_a_model_suggestion(agent):
    result = request_bandwidth(agent)
    assert result["allocated_bandwidth"] == 20
    commit(agent, result)
    assert agent.get_current_network_state()["embb_slice"]["resource_usage"] == 20


@pytest.mark.parametrize("cqi,occupied", [(1, False), (5, True)])
def test_infeasible_new_user_does_not_change_existing_allocations(agent, cqi, occupied):
    if occupied:
        agent.update_network_state(arrival_state())  # Existing minima 74 + new minimum 17 > 90.
    before = deepcopy(agent.get_current_network_state())
    result = request_bandwidth(agent, cqi)
    assert not result["admission_feasible"]
    assert result["allocated_bandwidth"] == 0
    agent.llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="Reviewed."))
    state = {
        "step_count": 0, "user_id": "21", "cqi": cqi,
        "request": "I want to watch 4K video", "history": [],
        "memory": {"final_slice": "eMBB", "balance_applied": True},
    }
    planned = agent.allocate_resources(state)
    assert planned["memory"]["allocation_failed"]
    assert agent.get_current_network_state() == before


def test_stale_preview_cannot_overwrite_newer_state(agent):
    result = request_bandwidth(agent)
    agent.update_network_state(arrival_state())
    before = agent.get_current_network_state()
    with pytest.raises(ValueError, match="Stale"):
        commit(agent, result)
    assert agent.get_current_network_state() == before
