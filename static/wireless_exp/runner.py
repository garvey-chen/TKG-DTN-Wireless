"""Sequential arrivals, isolated state per trial, and evaluator-only labels."""

import contextlib
import csv
import importlib.metadata
import json
import platform
import random
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import ROOT
from .data import load_users, scenario_path, sha256
from .llm import LoggedLLM
from .metrics import audit_state
from .policies import OriginalAgent, OriginalRule
from .settings import load_llm_settings, public_settings

METHODS = ("rule_upstream", "prompt", "wireless_kb", "wireless_nkb", "wireless_kb_guarded", "tkg_dtn")


def input_path(config: dict, scene: str) -> Path:
    if config.get("scenario_data_dir"):
        return Path(config["scenario_data_dir"]) / f"{scene}.csv"
    return scenario_path(scene)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes() -> dict:
    return {
        str(p.relative_to(ROOT)): sha256(p)
        for p in sorted((ROOT / "wireless_exp").rglob("*.py"))
    }


def write_steps_csv(path: Path, steps: list[dict]) -> None:
    if not steps:
        return
    rows = [{k: v for k, v in step.items() if k not in ("state", "result", "allocations", "violations")} for step in steps]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_trial(job: dict) -> dict:
    config = job["config"]
    scene, method, repeat = job["scene"], job["method"], job["repeat"]
    out = Path(job["output"]) / f"{scene}__{method}__r{repeat:03d}"
    out.mkdir(parents=True, exist_ok=False)
    path = input_path(config, scene)
    users = load_users(path, config["users"])
    # A repeat's order is paired across every method, never seeded by method.
    seed = config["seed"] + repeat - 1
    random.seed(seed)
    if config["arrival_order"] == "shuffle":
        random.shuffle(users)
    labels = {u.user_id: u.ground_truth for u in users}
    manifest = {
        "status": "running", "scenario": scene, "method": method, "repeat": repeat,
        "seed": seed, "arrival_order": config["arrival_order"],
        "user_ids": [u.user_id for u in users], "users_requested": len(users),
        "scenario_sha256": sha256(path), "scenario_file": str(path.resolve()),
        "knowledge_sha256": sha256(ROOT / "data/knowledge/Intent_Understand.txt"),
        "code_sha256": source_hashes(), "config": config,
        "python": platform.python_version(), "platform": platform.platform(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "semantic_mode": ("graph_react_exact_resources_incremental_v2" if method == "tkg_dtn" else
                          "upstream_kb_with_common_guard" if method == "wireless_kb_guarded" else
                          "wireless_agent_with_shared_exact_resources" if method in ("wireless_kb", "wireless_nkb") else
                          "upstream_algorithms_with_state_isolation_and_independent_audit"),
        "labels_exposed_to_policy": method == "rule_upstream",
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "pandas", "langgraph", "langchain-core", "requests")
        },
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "selected_users.json", [u.as_dict() for u in users])
    steps, llm, policy = [], None, None
    started = time.perf_counter()
    try:
        if method == "rule_upstream":
            policy = OriginalRule()
        else:
            credential_path = Path(job["credential_config"]) if job.get("credential_config") else None
            settings = load_llm_settings(config, credential_path)
            manifest["llm"] = public_settings(settings)
            write_json(out / "manifest.json", manifest)
            llm = LoggedLLM(settings, out / "llm_calls.jsonl")
            if method == "tkg_dtn":
                from .tkg.policy import TKGPolicy
                policy = TKGPolicy(llm, Path(config["neo4j_config"]), out,
                                   max_iterations=config.get("tkg_max_iterations", 8))
                manifest["graph_run_id"] = policy.graph.run_id
                manifest["graph_mapping_version"] = "incremental_v2"
                manifest["resource_tools_version"] = policy.tools.version
                manifest["execution_optimizations"] = policy.execution_optimizations
                manifest["graph_transport"] = policy.connection.transport
                manifest["intent_knowledge_context"] = policy.intent_knowledge_context
            elif method == "wireless_kb_guarded":
                from .tkg.executor import GuardedWirelessAgent
                policy = GuardedWirelessAgent(llm)
            else:
                policy = OriginalAgent(method, llm)
                if method in ("wireless_kb", "wireless_nkb"):
                    manifest["resource_tools_version"] = "exact_resources_v2"
        for index, user in enumerate(users, 1):
            step_started = time.perf_counter()
            # Redirect only within this process. Concurrent trials use processes.
            with (out / "upstream.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                result = policy.step_oracle(user) if method == "rule_upstream" else policy.step(user.observation())
            state = policy.snapshot()
            metrics = audit_state(state, labels, integer_bandwidth=method != "rule_upstream")
            selected_slice = result.get("slice_type")
            record = {
                "arrivals": index, "user_id": user.user_id,
                "intent_correct_this_arrival": selected_slice == user.ground_truth if selected_slice in ("eMBB", "URLLC") else None,
                "step_seconds": time.perf_counter() - step_started,
                **{k: v for k, v in metrics.items() if k not in ("allocations", "violations")},
                "valid_correct_admission_pct": metrics["valid_correct_users"] / index * 100,
                "llm_calls_cumulative": llm.calls if llm else 0,
                "violations": metrics["violations"], "allocations": metrics["allocations"],
                "state": state, "result": result,
            }
            steps.append(record)
            with (out / "steps.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            write_steps_csv(out / "steps.csv", steps)
            manifest["users_processed"] = index
            manifest["llm_statistics"] = llm.statistics() if llm else {}
            write_json(out / "manifest.json", manifest)
            print(f"[{scene}/{method}/r{repeat}] {index}/{len(users)} arrivals; "
                  f"admitted={metrics['admitted_users']}, valid+correct={metrics['valid_correct_users']}, "
                  f"audit_issues={metrics['violation_count']}", flush=True)
        manifest["status"] = "complete"
    except Exception as exc:
        # Transport sanitizes API errors. No config object/headers are logged.
        message = str(exc)
        if llm:
            message = message.replace(llm.settings["api_key"], "[REDACTED]")
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {message}"
        print(f"[{scene}/{method}] FAILED: {manifest['error']}", flush=True)
    finally:
        if policy is not None and hasattr(policy, "close"):
            try:
                policy.close()
            except Exception as exc:
                manifest["status"] = "failed"
                manifest["error"] = f"Graph export failed: {type(exc).__name__}"
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["users_processed"] = len(steps)
        manifest["llm_statistics"] = llm.statistics() if llm else {}
        write_json(out / "manifest.json", manifest)
        intent = [s["intent_correct_this_arrival"] for s in steps if s["intent_correct_this_arrival"] is not None]
        summary = {
            "status": manifest["status"], "method": method, "scenario": scene, "repeat": repeat,
            "users_processed": len(steps), "users_requested": len(users),
            "mean_utilization_over_all_processed_arrivals_pct": sum(s["bandwidth_utilization_pct"] for s in steps) / len(steps) if steps else None,
            "intent_accuracy_on_classified_arrivals_pct": sum(intent) / len(intent) * 100 if intent else None,
            "intent_classified_arrivals": len(intent),
            "mean_decision_seconds": sum(s["step_seconds"] for s in steps) / len(steps) if steps else None,
            "steps_with_audit_issues": sum(s["violation_count"] > 0 for s in steps),
            "final": {k: v for k, v in steps[-1].items() if k not in ("state", "result", "allocations", "violations")} if steps else None,
            "llm_statistics": manifest["llm_statistics"],
            "error": manifest.get("error"),
        }
        write_json(out / "summary.json", summary)
        if llm:
            llm.session.close()
    return {"directory": str(out), **summary}


def run_matrix(config: dict, output: Path | None, credential_config: Path | None, workers: int) -> Path:
    if "tkg_dtn" in config["methods"]:
        from .tkg.lifecycle import experiment_lock
        with experiment_lock():
            return _run_matrix(config, output, credential_config, workers)
    return _run_matrix(config, output, credential_config, workers)


def _run_matrix(config: dict, output: Path | None, credential_config: Path | None, workers: int) -> Path:
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "results" / f"{stamp}_{uuid.uuid4().hex[:6]}"
    output = output.resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new directory to preserve prior runs.")
    # Validate all data and credentials before starting any paid requests.
    for scene in config["scenarios"]:
        load_users(input_path(config, scene), config["users"])
    if any(m != "rule_upstream" for m in config["methods"]):
        load_llm_settings(config, credential_config)
    if "tkg_dtn" in config["methods"]:
        from .tkg.connection import GraphConnection
        graph = GraphConnection(Path(config["neo4j_config"]))
        try:
            if not graph.verify():
                raise RuntimeError("Neo4j is unavailable")
        finally:
            graph.close()
    if config["repeats"] < 1 or config["users"] < 1 or workers < 1:
        raise ValueError("users, repeats and workers must be positive")
    if not set(config["methods"]).issubset(METHODS):
        raise ValueError("Unknown method")
    output.mkdir(parents=True)
    if "tkg_dtn" in config["methods"]:
        from .tkg.lifecycle import reset_graph
        graph = GraphConnection(Path(config["neo4j_config"]))
        try:
            reset = reset_graph(graph, output / "pre_run_graph_backup", config.get("graph_reset", "database"))
            write_json(output / "graph_reset.json", reset)
            print(f"Graph reset: scope={reset['scope']}, deleted_nodes={reset['deleted_nodes']}; backup={output / 'pre_run_graph_backup'}", flush=True)
        finally:
            graph.close()
    write_json(output / "config.json", config)
    jobs = [
        {"config": config, "scene": scene, "method": method, "repeat": repeat,
         "output": str(output), "credential_config": str(credential_config.resolve()) if credential_config else None}
        for scene in config["scenarios"] for method in config["methods"]
        for repeat in range(1, config["repeats"] + 1)
    ]
    if workers == 1:
        summaries = [run_trial(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(run_trial, jobs))
    write_json(output / "matrix_summary.json", summaries)
    print(f"Results: {output}", flush=True)
    if any(s["status"] != "complete" for s in summaries):
        raise RuntimeError("One or more trials failed; inspect their manifest.json. Partial results are not full experiments.")
    return output
