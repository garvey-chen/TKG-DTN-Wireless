"""Neo4j transactions through Bolt, HTTP, or Windows localhost from WSL.

Only explicit local configuration is read. Credentials are never logged.
The Windows bridge passes credentials through stdin, not command arguments.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import requests

from .. import ROOT


class GraphConnection:
    def __init__(self, config_path: Path):
        self.settings = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("username", "password", "database"):
            self.settings[key] = os.environ.get("WIRELESS_NEO4J_" + key.upper(), self.settings.get(key, ""))
        if not self.settings.get("password"):
            raise ValueError("Fill password in the ignored configs/local.neo4j.json or WIRELESS_NEO4J_PASSWORD")
        self.transport = self.settings.get("transport", "bolt")
        self.calls = 0
        self.seconds = 0.0
        self.driver = None
        self.log_path = None
        self.session = requests.Session()
        if self.transport == "bolt":
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.settings["uri"], auth=(self.settings["username"], self.settings["password"]))
        elif self.transport == "windows_http":
            self.bridge_path = subprocess.check_output(
                ["wslpath", "-w", str(ROOT / "tools/windows_neo4j_bridge.ps1")], text=True
            ).strip()
        elif self.transport != "http":
            raise ValueError("Neo4j transport must be bolt, http or windows_http")

    def public_settings(self):
        return {k: v for k, v in self.settings.items() if k in ("transport", "http_url", "uri", "database")}

    def query(self, query: str, parameters: dict | None = None) -> list[dict]:
        record = {"cypher": query, "parameters": parameters or {}}
        started = time.perf_counter()
        try:
            rows = self._execute(query, parameters)
            record["rows"] = rows
            return rows
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            if self.log_path:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _execute(self, query: str, parameters: dict | None = None) -> list[dict]:
        started = time.perf_counter()
        self.calls += 1
        try:
            if self.driver:
                from neo4j import Query
                with self.driver.session(database=self.settings["database"]) as session:
                    return session.run(Query(query, timeout=30), parameters or {}).data()
            payload = {"statements": [{"statement": query, "parameters": parameters or {}, "resultDataContents": ["row"]}]}
            url = self.settings["http_url"].rstrip("/") + "/db/" + quote(self.settings["database"], safe="") + "/tx/commit"
            if self.transport == "windows_http":
                request = {"url": url, "username": self.settings["username"], "password": self.settings["password"], **payload}
                proc = subprocess.run([
                    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                    "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", self.bridge_path,
                ], input=json.dumps(request, ensure_ascii=False), text=True, encoding="utf-8", capture_output=True, timeout=45)
                try:
                    data = json.loads(proc.stdout.lstrip("\ufeff"))
                except ValueError:
                    raise RuntimeError("Windows Neo4j bridge did not return JSON") from None
                if proc.returncode or data.get("bridge_error"):
                    raise RuntimeError(f"Neo4j bridge HTTP {data.get('status')}; codes={data.get('codes', [])}")
            else:
                response = self.session.post(url, json=payload, auth=(self.settings["username"], self.settings["password"]), timeout=35)
                if not response.ok:
                    raise RuntimeError(f"Neo4j HTTP {response.status_code}; body omitted")
                data = response.json()
            if data.get("errors"):
                # Server error text may echo parameters; only publish error codes.
                raise RuntimeError("Neo4j query failed: " + ", ".join(e["code"] for e in data["errors"]))
            result = data["results"][0]
            return [dict(zip(result["columns"], row["row"])) for row in result["data"]]
        except (requests.RequestException, subprocess.SubprocessError):
            raise RuntimeError("Neo4j transport unavailable; no simulated graph substituted") from None
        finally:
            self.seconds += time.perf_counter() - started

    def verify(self):
        return self.query("RETURN 1 AS ok") == [{"ok": 1}]

    def close(self):
        self.session.close()
        if self.driver:
            self.driver.close()
