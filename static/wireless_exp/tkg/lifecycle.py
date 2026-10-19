"""One cleanup per experiment matrix, before worker processes start.

The workspace lock prevents one CLI matrix from deleting another matrix's graph.
Backups contain graph data only, never connection credentials. Indexes, database
configuration and authentication are not deleted.
"""

import json
import os
from contextlib import contextmanager

from .. import ROOT


@contextmanager
def experiment_lock():
    directory = ROOT / "runtime"
    directory.mkdir(exist_ok=True)
    with (directory / "neo4j_experiment.lock").open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another wireless matrix is running; cleanup refused to protect its graph") from None
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def reset_graph(connection, backup_directory, scope="database"):
    """Export before deletion. Default full-database reset is user-authorized.

    'wireless' removes this adapter's nodes only; 'none' preserves prior runs.
    All branches are called once by the matrix parent, never by trial workers.
    """
    if scope not in ("database", "wireless", "none"):
        raise ValueError("graph_reset must be database, wireless, or none")
    if scope == "none":
        return {"scope": scope, "deleted_nodes": 0}
    backup_directory.mkdir(parents=True, exist_ok=False)
    condition = "true" if scope == "database" else "(n:WirelessEntity OR n:WirelessRun) AND n.run_id STARTS WITH 'wireless_'"
    # Back up the entire graph even for a selective reset, preserving endpoints
    # and incoming relationships that a DETACH DELETE would otherwise remove.
    counts = connection.query("MATCH (n) RETURN count(n) AS nodes")[0]
    nodes_file = backup_directory / "nodes.jsonl"
    edges_file = backup_directory / "relationships.jsonl"
    node_count = edge_count = 0
    # Stream pages to disk. Any read/write failure occurs before deletion.
    for file, query in (
        (nodes_file, "MATCH (n) RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS properties ORDER BY element_id SKIP $skip LIMIT $limit"),
        (edges_file, "MATCH (n)-[r]->(b) RETURN elementId(r) AS element_id, elementId(n) AS source, elementId(b) AS target, type(r) AS type, properties(r) AS properties ORDER BY element_id SKIP $skip LIMIT $limit"),
    ):
        with file.open("w", encoding="utf-8") as handle:
            offset = 0
            while True:
                rows = connection.query(query, {"skip": offset, "limit": 2000})
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                offset += len(rows)
                if len(rows) < 2000:
                    break
            handle.flush()
            os.fsync(handle.fileno())
        if file == nodes_file:
            node_count = offset
        else:
            edge_count = offset
    if node_count != counts["nodes"]:
        raise RuntimeError("Graph changed during backup; reset not attempted")
    metadata = {"scope": scope, "connection": connection.public_settings(),
                "backup_nodes": node_count, "backup_relationships": edge_count,
                "status": "backup_complete"}
    manifest = backup_directory / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    deleted = connection.query(f"""
MATCH (n) WHERE {condition}
WITH collect(n) AS nodes
FOREACH (node IN nodes | DETACH DELETE node)
RETURN size(nodes) AS deleted_nodes
""")[0]["deleted_nodes"]
    remaining = connection.query(f"MATCH (n) WHERE {condition} RETURN count(n) AS nodes")[0]["nodes"]
    metadata.update(status="cleared" if remaining == 0 else "unexpected_remaining_data", deleted_nodes=deleted, remaining_nodes=remaining)
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if remaining:
        raise RuntimeError("Graph received new data during reset; experiment not started")
    return metadata
