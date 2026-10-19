"""Read-only validation: no model API calls, no changes to supplied datasets."""

import ast
import importlib.util
import json
import xml.etree.ElementTree as ET

from . import ROOT
from .data import load_users, scenario_path, sha256
from .policies import load_legacy


def preflight() -> dict:
    dependencies = ("numpy", "scipy", "pandas", "matplotlib", "langgraph", "langchain_core", "requests", "tabulate", "shapely", "pyproj", "yaml")
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing dependencies: " + ", ".join(missing))
    files = json.loads((ROOT / "vendor/manifest.json").read_text(encoding="utf-8"))["files"]
    for item in files:
        path = ROOT / "vendor/upstream" / item["path"]
        if sha256(path) != item["snapshot_sha256"]:
            raise ValueError(f"Modified upstream snapshot: {item['path']}")
    scenes = {}
    for scene in ("center", "north", "south"):
        path = scenario_path(scene)
        users = load_users(path)
        source = ROOT / "vendor/upstream/Knowledge_Base" / f"ray_tracing_results_{scene}.csv"
        if sha256(path) != sha256(source):
            raise ValueError(f"Working dataset differs from upstream: {scene}")
        map_path = ROOT / "data/maps" / f"HKUST_{scene.title()}.osm"
        tree = ET.parse(map_path).getroot()
        buildings = [way for way in tree.findall("way") if any(t.get("k") == "building" for t in way.findall("tag"))]
        scenes[scene] = {"rows": len(users), "first30": len(users[:30]), "building_outlines": len(buildings), "sha256": sha256(path)}
    for path in (ROOT / "wireless_exp").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    # Load imports and compile graphs without any API invocation.
    for name in ("agent_kb", "agent_nkb"):
        module = load_legacy(name)
        module.create_network_graph()
    load_legacy("prompt")
    load_legacy("rule")
    return {"status": "passed", "scenarios": scenes, "verified_upstream_files": len(files), "api_calls": 0}
