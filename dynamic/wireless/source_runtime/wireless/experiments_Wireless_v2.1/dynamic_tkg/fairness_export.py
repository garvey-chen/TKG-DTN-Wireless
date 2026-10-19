"""Read-only JSON/CSV export of the additive fairness graph schema."""

from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3

from .fairness_schema import NODE_TYPES, RELATION_TYPES, mermaid


def export_graph(database, output):
    database, output = Path(database).resolve(), Path(output)
    db = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("BEGIN")
        metadata = {r["key"]: json.loads(r["value_json"]) for r in db.execute("SELECT * FROM metadata")}
        if "fairness_config" not in metadata:
            raise ValueError("Expected an explicitly configured fairness graph")
        nodes = [{"id": r["id"], "kind": r["kind"], "scenario": r["scenario"],
                  "props": json.loads(r["properties_json"])} for r in db.execute("SELECT * FROM nodes ORDER BY id")]
        edges = [{"id": r["id"], "type": r["relation"], "scenario": r["scenario"],
                  "source": r["source"], "target": r["target"], "props": json.loads(r["properties_json"])}
                 for r in db.execute("SELECT * FROM edges ORDER BY id")]
    finally:
        db.close()
    lookup, run_id = {n["id"]: n for n in nodes}, metadata["run_id"]
    for node in nodes:
        if node["kind"] not in NODE_TYPES or node["props"].get("run_id") != run_id:
            raise ValueError("Invalid node type or run provenance")
    for edge in edges:
        if edge["type"] not in RELATION_TYPES or edge["props"].get("run_id") != run_id:
            raise ValueError("Invalid relationship or run provenance")
        if edge["source"] not in lookup or edge["target"] not in lookup:
            raise ValueError("Missing relationship endpoint")
        source, target = RELATION_TYPES[edge["type"]]
        if (source != "*" and lookup[edge["source"]]["kind"] != source) or lookup[edge["target"]]["kind"] != target:
            raise ValueError("Relationship violates fairness metagraph")
    schema = metadata["schema"]
    if schema["node_types"] != NODE_TYPES or {k: tuple(v) for k, v in schema["relationships"].items()} != RELATION_TYPES:
        raise ValueError("Stored graph schema does not match the export schema")
    graph = {"run_id": run_id, "backend_source": "local_sqlite", "metadata": metadata,
             "nodes": nodes, "edges": edges}
    serialized = json.dumps(graph, ensure_ascii=False, sort_keys=True, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / "graph.json").write_text(serialized + "\n", encoding="utf-8")
    (output / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "schema.mmd").write_text(mermaid(), encoding="utf-8")
    for name, rows, fields in (("nodes", nodes, ["id", "kind", "scenario"]),
                               ("edges", edges, ["id", "type", "scenario", "source", "target"])):
        with (output / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields + ["properties_json"])
            writer.writeheader()
            writer.writerows({**{k: row[k] for k in fields}, "properties_json": json.dumps(row["props"], ensure_ascii=False,
                              sort_keys=True, allow_nan=False)} for row in rows)
    return {"run_id": run_id, "node_count": len(nodes), "edge_count": len(edges),
            "node_counts_by_kind": dict(sorted(Counter(n["kind"] for n in nodes).items())),
            "edge_counts_by_type": dict(sorted(Counter(e["type"] for e in edges).items())),
            "backend_source": "local_sqlite", "neo4j_import_executed": False,
            "artifacts": {name: str(output / name) for name in
                          ("graph.json", "nodes.csv", "edges.csv", "schema.json", "schema.mmd")}}
