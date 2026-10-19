#!/usr/bin/env python3
"""Portable command line entry point; paths resolve relative to this project."""

import argparse
import json
from pathlib import Path

from wireless_exp.settings import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Validate data, dependencies and graphs without LLM calls")
    run = sub.add_parser("run", help="Execute original policies with logged real LLM calls")
    run.add_argument("--config", type=Path)
    run.add_argument("--methods", nargs="+", choices=("rule_upstream", "prompt", "wireless_kb", "wireless_nkb", "wireless_kb_guarded", "tkg_dtn"), default=["rule_upstream"])
    run.add_argument("--scenarios", nargs="+", choices=("center", "north", "south"))
    run.add_argument("--users", type=int)
    run.add_argument("--repeats", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--arrival-order", choices=("source", "shuffle"))
    run.add_argument("--credential-config", type=Path, help="Optional existing YAML containing llm.api_key; never copied")
    run.add_argument("--neo4j-config", type=Path, help="Ignored local JSON containing Neo4j connection settings")
    run.add_argument("--graph-reset", choices=("database", "wireless", "none"), default=None,
                     help="Before a TKG matrix: backup then reset database (default), wireless nodes only, or preserve data")
    run.add_argument("--output", type=Path)
    run.add_argument("--scenario-data-dir", type=Path, help="Optional generated CSV directory; defaults to supplied datasets")
    run.add_argument("--workers", type=int, default=1, help="Independent trial processes (default 1)")
    plot = sub.add_parser("report", help="Plot complete saved trials")
    plot.add_argument("directory", type=Path)
    generate = sub.add_parser("generate", help="Generate NEW users/channels from the included maps")
    generate.add_argument("--scenarios", nargs="+", choices=("center", "north", "south"), default=["center", "north", "south"])
    generate.add_argument("--users", type=int, default=30)
    generate.add_argument("--seed", type=int, default=20261008)
    generate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        from wireless_exp.preflight import preflight
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
    elif args.command == "report":
        from wireless_exp.report import report
        print(json.dumps(report(args.directory.resolve()), indent=2))
    elif args.command == "generate":
        from wireless_exp.scene import generate_scene
        for scene in args.scenarios:
            print(generate_scene(scene, args.users, args.seed, args.output.resolve()))
    else:
        from wireless_exp.runner import run_matrix
        config = load_config(args.config)
        for name in ("methods", "scenarios", "users", "repeats", "seed", "arrival_order"):
            value = getattr(args, name)
            if value is not None:
                config[name] = value
        if args.scenario_data_dir:
            config["scenario_data_dir"] = str(args.scenario_data_dir.resolve())
        if args.neo4j_config:
            config["neo4j_config"] = str(args.neo4j_config.resolve())
        elif "tkg_dtn" in config["methods"]:
            from wireless_exp import ROOT
            config["neo4j_config"] = str(ROOT / "configs/local.neo4j.json")
        if "tkg_dtn" in config["methods"]:
            config["graph_reset"] = args.graph_reset or config.get("graph_reset", "database")
        run_matrix(config, args.output, args.credential_config, args.workers)


if __name__ == "__main__":
    main()
