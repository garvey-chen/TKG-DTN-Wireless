"""Export the actual local SQLite graph; never connect to or clear Neo4j."""

import csv
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

from .schema import NODE_TYPES, RELATION_TYPES, definition, mermaid


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _cypher_string(value):
    """Quote arbitrary text as a Cypher single-quoted string literal.

    JSON object syntax is not valid Cypher map syntax. In particular, quoted
    property keys and double-quote escaping cannot simply be copied wholesale.
    """
    escaped = []
    codes = {"\\": "\\\\", "'": "\\'", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}
    for char in value:
        if char in codes:
            escaped.append(codes[char])
        elif ord(char) < 32 or ord(char) == 127:
            escaped.append(f"\\u{ord(char):04x}")
        else:
            escaped.append(char)
    return "'" + "".join(escaped) + "'"


def _cypher(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _cypher_string(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Cannot export nonfinite graph properties")
        return repr(value)
    if isinstance(value, list):
        return "[" + ",".join(_cypher(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join("`" + str(key).replace("`", "``") + "`:" + _cypher(item)
                               for key, item in value.items()) + "}"
    raise TypeError(f"Unsupported Cypher value: {type(value).__name__}")


def _properties(props, **canonical):
    """Keep native scalars queryable and encode nested values losslessly."""
    result = {key: value for key, value in props.items() if not isinstance(value, (list, dict))}
    for key, value in props.items():
        if isinstance(value, (list, dict)) and key + "_json" not in props:
            result[key + "_json"] = _json(value)
    result["properties_json"] = _json(props)
    result.update(canonical)
    return result


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _query_examples():
    return """// Set $rid to the run_id in graph.json before using these read-only queries.
// Nodes always retain complete source properties in properties_json.

// Latest user projections and optional current allocations.
MATCH (u:WirelessUser {run_id:$rid})
OPTIONAL MATCH (u)-[:CURRENT_ALLOCATION]->(a:WirelessAllocation {run_id:$rid})
RETURN u.scenario AS scenario, u.user_id AS user_id,
       properties(u) AS user, properties(a) AS allocation
ORDER BY scenario, user_id;

// Service episode history and the evidence attached to episode events.
MATCH (u:WirelessUser {run_id:$rid})-[:HAS_EPISODE]->(ep:WirelessServiceEpisode {run_id:$rid})
OPTIONAL MATCH (ev:WirelessServiceEvent {run_id:$rid})-[:IN_EPISODE]->(ep)
OPTIONAL MATCH (ev)-[:EVIDENCED_BY]->(o:WirelessObservation {run_id:$rid})
RETURN u.scenario AS scenario, u.user_id AS user_id,
       properties(ep) AS episode, properties(ev) AS event, properties(o) AS evidence
ORDER BY scenario, user_id, ev.time_s;

// One frame's historical channel and service status (set $scenario and $time_s).
MATCH (f:WirelessFrame {run_id:$rid,scenario:$scenario,time_s:$time_s})
MATCH (u:WirelessUser {run_id:$rid})-[:HAS_OBSERVATION]->(o:WirelessObservation {run_id:$rid})-[:IN_FRAME]->(f)
OPTIONAL MATCH (o)-[:HAS_SERVICE_STATUS]->(s:WirelessServiceStatus {run_id:$rid})
RETURN f.time_s AS time_s, u.user_id AS user_id,
       properties(o) AS observation, properties(s) AS service_status
ORDER BY user_id;
"""


def export_graph(database: Path, output: Path) -> dict:
    """Export one consistent SQLite snapshot and a standalone Cypher import.

    The generated script performs scoped upserts only, intended for a fresh
    run_id in Neo4j. It never deletes existing data. Reimporting the same snapshot
    is idempotent; importing older/different snapshots over an existing run is
    not a historical restore and can leave relationships from that existing run.
    """
    database, output = Path(database).resolve(), Path(output)
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        row = connection.execute("SELECT value_json FROM metadata WHERE key='run_id'").fetchone()
        if row is None:
            raise ValueError("Graph metadata is missing run_id")
        run_id = json.loads(row["value_json"])
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Graph run_id must be a nonempty string")
        nodes = [{"id": row["id"], "kind": row["kind"], "scenario": row["scenario"],
                  "props": json.loads(row["properties_json"])}
                 for row in connection.execute("SELECT * FROM nodes ORDER BY id")]
        edges = [{"id": row["id"], "type": row["relation"], "scenario": row["scenario"],
                  "source": row["source"], "target": row["target"],
                  "props": json.loads(row["properties_json"])}
                 for row in connection.execute("SELECT * FROM edges ORDER BY id")]
    finally:
        connection.close()

    node_by_id = {node["id"]: node for node in nodes}
    for node in nodes:
        if node["kind"] not in NODE_TYPES:
            raise ValueError(f"Unknown node kind: {node['kind']}")
        if not isinstance(node["props"], dict) or node["props"].get("run_id") != run_id:
            raise ValueError(f"Node does not belong to exported run: {node['id']}")
    for edge in edges:
        if edge["type"] not in RELATION_TYPES:
            raise ValueError(f"Unknown relationship: {edge['type']}")
        if not isinstance(edge["props"], dict) or edge["props"].get("run_id") != run_id:
            raise ValueError(f"Edge does not belong to exported run: {edge['id']}")
        if edge["source"] not in node_by_id or edge["target"] not in node_by_id:
            raise ValueError(f"Missing endpoint for edge: {edge['id']}")
        source_type, target_type = RELATION_TYPES[edge["type"]]
        if (source_type != "*" and node_by_id[edge["source"]]["kind"] != source_type
                or node_by_id[edge["target"]]["kind"] != target_type):
            raise ValueError(f"Relationship violates metagraph: {edge['id']}")

    graph = {"run_id": run_id, "backend_source": "local_sqlite", "nodes": nodes, "edges": edges}
    serialized = _json(graph)  # Validate all numeric/property values before writing outputs.
    output.mkdir(parents=True, exist_ok=True)
    (output / "graph.json").write_text(serialized + "\n", encoding="utf-8")
    (output / "schema.json").write_text(json.dumps(definition(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "schema.mmd").write_text(mermaid(), encoding="utf-8")
    _write_csv(output / "nodes.csv", ["id", "kind", "scenario", "properties_json"],
               ({"id": node["id"], "kind": node["kind"], "scenario": node["scenario"],
                 "properties_json": _json(node["props"])} for node in nodes))
    _write_csv(output / "edges.csv", ["id", "type", "scenario", "source", "target", "properties_json"],
               ({"id": edge["id"], "type": edge["type"], "scenario": edge["scenario"],
                 "source": edge["source"], "target": edge["target"],
                 "properties_json": _json(edge["props"])} for edge in edges))

    with (output / "neo4j_import.cypher").open("w", encoding="utf-8") as handle:
        handle.write("// Offline export only: this file has not been executed against Neo4j.\n")
        handle.write("// Import into a fresh run_id. Scoped MERGE only; no database reset or deletion.\n")
        handle.write("// This is a snapshot upsert, not a restore over another snapshot of the same run.\n")
        handle.write("// Nested properties use *_json; complete original properties remain in properties_json.\n")
        run_literal = _cypher(run_id)
        handle.write(f"MERGE (r:WirelessRun {{run_id:{run_literal}}}) "
                     "SET r.backend_source='local_sqlite_export', r.schema_version='wireless_dynamic_v2.1';\n")
        # Whitelisted labels and relationship types are compiled in this module;
        # user text is encoded only as values, never as executable identifiers.
        for kind in NODE_TYPES:
            selected = [node for node in nodes if node["kind"] == kind]
            for offset in range(0, len(selected), 250):
                rows = [{"id": node["id"], "props": _properties(node["props"], id=node["id"],
                         kind=kind, scenario=node["scenario"], run_id=run_id)}
                        for node in selected[offset:offset + 250]]
                handle.write(f"UNWIND {_cypher(rows)} AS row\n"
                             f"MERGE (n:WirelessEntity {{run_id:{run_literal}, id:row.id}})\n"
                             f"SET n:Wireless{kind}, n += row.props;\n")
        for relation in RELATION_TYPES:
            selected = [edge for edge in edges if edge["type"] == relation]
            for offset in range(0, len(selected), 250):
                rows = [{"id": edge["id"], "source": edge["source"], "target": edge["target"],
                         "props": _properties(edge["props"], id=edge["id"],
                                              scenario=edge["scenario"], run_id=run_id)}
                        for edge in selected[offset:offset + 250]]
                handle.write(f"UNWIND {_cypher(rows)} AS row\n"
                             f"MATCH (a:WirelessEntity {{run_id:{run_literal}, id:row.source}}), "
                             f"(b:WirelessEntity {{run_id:{run_literal}, id:row.target}})\n"
                             f"MERGE (a)-[r:{relation} {{run_id:{run_literal}, id:row.id}}]->(b)\n"
                             "SET r += row.props;\n")
    (output / "query_examples.cypher").write_text(_query_examples(), encoding="utf-8")
    return {
        "run_id": run_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_counts_by_kind": dict(sorted(Counter(node["kind"] for node in nodes).items())),
        "edge_counts_by_type": dict(sorted(Counter(edge["type"] for edge in edges).items())),
        "backend_source": "local_sqlite",
        "neo4j_import_executed": False,
        "artifacts": {name: str(output / name) for name in (
            "graph.json", "nodes.csv", "edges.csv", "schema.json", "schema.mmd",
            "neo4j_import.cypher", "query_examples.cypher")},
    }
