"""Regenerate local adaptations and reviewable diffs from the sanitized snapshot.

Run with: python -m wireless_exp.adapt_sources
This edits only wireless_exp/legacy and docs/patches inside this project.
"""

import ast
import difflib
import re

from . import ROOT

SOURCES = {
    "agent_kb.py": "WA_DS_V3_KB.py",
    "agent_nkb.py": "WA_DS_V3_NKB.py",
    "prompt.py": "Thro_NetwScen_Methods/Prompt_Based_Center.py",
    "rule.py": "Thro_NetwScen_Methods/Optimal_RA_Center.py",
    "propagation.py": "RayTracing_cqi.py",
}


SHARED_BEAMFORMING = '''
def beamforming_tool(user_id: str, slice_type: str, cqi: int, request: str) -> Dict[str, Any]:
    """Preview exact joint integer bandwidth allocation for new and existing users."""
    current_state = get_current_network_state()
    limits = SLICE_LIMITS[slice_type]
    latency = apply_heuristic_latency(slice_type, request, *limits["latency"])
    plan = plan_admission(current_state, user_id, cqi, slice_type, latency)
    certificate = plan["certificate"]
    allocation = plan.get("allocation")
    bandwidth = allocation["bandwidth"] if allocation else 0
    rate = allocation["rate"] if allocation else 0.0
    return {
        "status": "success" if plan["feasible"] else "warning",
        "user_id": user_id, "slice_type": slice_type, "cqi": cqi,
        "allocated_bandwidth": bandwidth, "allocated_rate": rate,
        "allocated_latency": latency, "rate_valid": plan["feasible"],
        "admission_feasible": plan["feasible"],
        "adjustment_made": bool(plan["adjustments"]),
        "adjustment_reason": "Joint allocation satisfies minima, then maximizes total rate for retained users."
            if plan["feasible"] else "No feasible integer admission while retaining existing users.",
        "rate_requirements": "100-400 Mbps" if slice_type == "eMBB" else "1-100 Mbps",
        "has_capacity": plan["feasible"],
        "available_bandwidth": certificate["available_before_adjustment_mhz"],
        "adjustment_needed": max(0, bandwidth - certificate["available_before_adjustment_mhz"]),
        "resource_plan": plan,
    }
'''

SHARED_COMMIT = '''
def slice_allocation(user_id: str, slice_type: str, rate: float, latency: float,
                     cqi: int, bandwidth: int, resource_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Commit the joint allocation preview without an additional model call."""
    allocation = resource_plan.get("allocation", {})
    if (resource_plan["user_id"] != str(user_id) or resource_plan["slice_type"] != slice_type
            or allocation.get("bandwidth") != bandwidth or allocation.get("cqi") != cqi
            or allocation.get("rate") != rate or allocation.get("latency") != latency):
        raise ValueError("Allocation arguments do not match the resource preview")
    candidate = commit_admission(get_current_network_state(), resource_plan)
    candidate["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_network_state(candidate)
    part = candidate[SLICE_LIMITS[slice_type]["key"]]
    return {
        "status": "success", "message": f"User {user_id} has been allocated to {slice_type} slice",
        "allocation": deepcopy(allocation) | {"slice_type": slice_type},
        "resource_engine": resource_plan["resource_engine"],
        "updated_slice": {"users_count": len(part["users"]),
                          "resource_usage": part["resource_usage"],
                          "utilization_rate": part["utilization_rate"]},
    }
'''

SHARED_ADJUSTMENT = '''    # The shared solver previews all reallocations; nothing changes before commit.
    resource_plan = beamforming_result["resource_plan"]
    adjustments = resource_plan["adjustments"]
    state["memory"]["adjustment_result"] = {
        "has_capacity": resource_plan["feasible"],
        "adjustments_made": bool(adjustments),
        "user_adjustments": adjustments,
        "freed_bandwidth": sum(a["bandwidth_reduction"] for a in adjustments),
    }
    adjustment_message = (
        "A feasible joint allocation is ready. Existing users and slices are retained; "
        "minimum service is satisfied, then remaining bandwidth maximizes total rate. "
        f"Existing users to reallocate: {len(adjustments)}."
        if resource_plan["feasible"] else
        "No feasible integer admission while retaining all existing users and their minimum service."
    )

'''


def replace_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [replacement.rstrip() + "\n"]
    return "".join(lines)


def adapt(name: str, text: str) -> str:
    header = (
        ("# Adapted locally from vendor/upstream. See docs/patches; shared exact bandwidth allocation.\n"
         if name.startswith("agent") else
         "# Adapted locally from vendor/upstream. See docs/patches; algorithm retained.\n")
        +
        "from copy import deepcopy\n"
        "from wireless_exp import ROOT as EXPERIMENT_ROOT\n"
    )
    if name.startswith("agent"):
        header += ("from wireless_exp.metrics import SLICE_LIMITS\n"
                   "from wireless_exp.shared_resources import plan_admission, commit_admission\n")
        text = text.replace("from langchain_openai import ChatOpenAI\n", "")
        text = re.sub(r"llm = ChatOpenAI\([\s\S]*?\n\)", "llm = None  # Injected by the experiment adapter.", text, count=1)
        text = replace_function(text, "beamforming_tool", SHARED_BEAMFORMING)
        text = replace_function(text, "slice_allocation", SHARED_COMMIT)
        start = text.index("    # Check if we have capacity or need adjustment", text.index("def allocate_resources("))
        end = text.index("    # Create resource allocation summary", start)
        text = text[:start] + SHARED_ADJUSTMENT + text[end:]
        needle = '        "bandwidth": beamforming_result["allocated_bandwidth"]\n'
        assert text.count(needle) == 1
        text = text.replace(needle, needle.rstrip() + ',\n        "resource_plan": beamforming_result["resource_plan"]\n', 1)
        text = text.replace("users had their bandwidth reduced to accommodate this new user", "existing users had their bandwidth reallocated by the shared optimizer")
        text = text.replace("Total bandwidth freed:", "Net bandwidth released from existing users:")
        text = text.replace("Bandwidth freed:", "Net bandwidth released:")
        text = text.replace("{adj['bandwidth_reduction']} MHz reduction", "{adj['bandwidth_change']:+} MHz change")
        if name == "agent_kb.py":
            text = replace_function(text, "load_knowledge_base", '''
def load_knowledge_base(file_path):
    """Fail explicitly on missing knowledge; never silently substitute test data."""
    with open(file_path, "r", encoding="utf-8") as handle:
        return handle.read()
''')
    if name == "prompt.py":
        text = replace_function(text, "call_llm", '''
def call_llm(prompt):
    """Transport is injected by the runner; original SYSTEM_PROMPT is retained."""
    raise RuntimeError("Use python run.py run to configure the LLM transport.")
''')
    # Make every legacy file path local. The runner selects datasets explicitly.
    text = re.sub(
        r'r"C:\\[^"\n]*\\(Intent_Understand\.txt|ray_tracing_results[^"\\]*\.csv|HKUST_F\.osm)"',
        lambda m: {
            "Intent_Understand.txt": 'str(EXPERIMENT_ROOT / "data/knowledge/Intent_Understand.txt")',
            "HKUST_F.osm": 'str(EXPERIMENT_ROOT / "data/maps/HKUST_F.osm")',
        }.get(m[1], 'str(EXPERIMENT_ROOT / "data/scenarios/center.csv")'),
        text,
    )
    if name.startswith("agent") or name == "prompt.py":
        text = text.replace("GLOBAL_NETWORK_STATE.copy()", "deepcopy(GLOBAL_NETWORK_STATE)")
        text = text.replace("INITIAL_NETWORK_STATE.copy()", "deepcopy(INITIAL_NETWORK_STATE)")
        text = text.replace("GLOBAL_NETWORK_STATE = new_state", "GLOBAL_NETWORK_STATE = deepcopy(new_state)")
    # Old main() functions remain for provenance but cannot accidentally run.
    text = re.sub(
        r'if __name__ == ["\']__main__["\']:[\s\S]*$',
        'if __name__ == "__main__":\n    raise SystemExit("Use the project run.py entry point.")\n',
        text,
    )
    if name.startswith("agent") or name == "prompt.py":
        # The project runner owns reset-once-per-trial and never calls old main().
        text = replace_function(text, "main", '''
def main(*args, **kwargs):
    """The runner owns dataset selection, trial reset and consistent metrics."""
    raise RuntimeError("Use the project run.py entry point.")
''')
    ast.parse(header + text)
    return header + text


def main():
    for name, source in SOURCES.items():
        original = (ROOT / "vendor/upstream" / source).read_text(encoding="utf-8")
        adapted = adapt(name, original)
        (ROOT / "wireless_exp/legacy" / name).write_text(adapted, encoding="utf-8")
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True), adapted.splitlines(keepends=True),
            fromfile="vendor/upstream/" + source, tofile="wireless_exp/legacy/" + name,
        ))
        (ROOT / "docs/patches" / (name + ".diff")).write_text(diff, encoding="utf-8")
        print(f"Adapted {source} -> legacy/{name}")


if __name__ == "__main__":
    main()
