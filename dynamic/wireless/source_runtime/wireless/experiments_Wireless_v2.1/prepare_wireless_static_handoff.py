"""Export WirelessAgent's own r001 terminal, with exact physical equivalence checks."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from continuation.export_static_terminal import (
    allocation_signature, physical_state_signature, source_record, terminal_step,
)
from continuation.handoff import load_handoff
from run_dynamic_graph import ROOT, SCENES, sha256, write_json


def prepare(source, batch, output):
    if output.exists():
        raise ValueError(f"Refusing to overwrite: {output}")
    verified, _ = load_handoff(source)
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    metadata = deepcopy(original)
    metadata.update(created_at=datetime.now(timezone.utc).isoformat(),
        purpose="Exact WirelessAgent KB static r001 terminal continuation; no admission or t0 optimization.",
        exporter=source_record(Path(__file__)),
        reference_handoff_manifest=source_record(source / "manifest.json"))
    prepared = []
    for scene in SCENES:
        trial = batch / f"{scene}__wireless_kb__r001"
        step, line = terminal_step(trial / "steps.jsonl")
        reference = verified[scene]["terminal_step"]
        if (allocation_signature(step) != allocation_signature(reference)
                or physical_state_signature(step) != physical_state_signature(reference)):
            raise ValueError(f"{scene}: static physical state differs; require a new paired design")
        selected = json.loads((trial / "selected_users.json").read_text(encoding="utf-8"))
        if selected != json.loads((source / scene / "selected_users.json").read_text(encoding="utf-8")):
            raise ValueError(f"{scene}: static population differs")
        info = next(item for item in metadata["scenes"] if item["scene"] == scene)
        info.update(source_method="wireless_kb", source_repeat=1, terminal_line_1based=line,
                    exact_physical_equivalence_to_tkg_r001=True)
        info["sources"] = {name: source_record(trial / filename) for name, filename in
            (("steps", "steps.jsonl"), ("selected_users", "selected_users.json"), ("summary", "summary.json"))}
        prepared.append((scene, trial, step))
    shutil.copytree(source, output)
    for scene, trial, step in prepared:
        write_json(output / scene / "terminal_step.json", step)
        write_json(output / scene / "raw_static_state.json", step["state"])
        shutil.copyfile(trial / "selected_users.json", output / scene / "selected_users.json")
    metadata["output_files"] = {str(p.relative_to(output)): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    write_json(output / "manifest.json", metadata)
    load_handoff(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "data/static_terminal_handoff_20261014")
    parser.add_argument("--static-batch", type=Path, default=ROOT.parent /
        "experiments_Wireless_v2/results/four_methods_5repeats_20261012_212809")
    parser.add_argument("--output", type=Path, default=ROOT / "data/wireless_static_terminal_handoff_20261015")
    args = parser.parse_args()
    prepare(args.reference.resolve(), args.static_batch.resolve(), args.output.resolve())
    print(f"Verified own WirelessAgent static terminals: {args.output}")


if __name__ == "__main__":
    main()
