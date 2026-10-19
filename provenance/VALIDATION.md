# Release validation — 19 October 2026

Validation ran under Python 3.11.15 in a separate directory outside the original workspace. No model API calls were made.

| Check | Result |
| --- | --- |
| Static preflight and upstream data hashes | Passed |
| Archived static trials | All 60 passed; 1,800 arrivals recomputed |
| Paired TKG-DTN/WirelessAgent resource states | All 450 identical |
| Archived dynamic trials | All 33 passed |
| Dynamic trial metrics and Table I | Recomputed values match manuscript CSVs |
| Within-slice sliding-window fairness | All 19,800 run/scenario/cohort/window values match |
| Initial handoff and membership | Identical across compared methods/repeats |
| 1 s mobility regeneration | All three scenario CSVs byte-identical to paper inputs |
| Static unit/semantic tests | 32 passed; 10 live Neo4j tests skipped |
| Completion scheduler tests | Both archived harnesses passed; synthetic durations, zero API calls |
| Dynamic runner preflight | TKG-DTN and WirelessAgent passed; zero API calls |
| Fresh deterministic frozen Center run | Complete; final metrics match recorded baseline |
| Fresh rule-based Center run | 30 arrivals completed |
| Credential scan | No suspected live credentials; matches are documented placeholders |

The fresh rule baseline is a CLI smoke check, not a replacement for the recorded five-repeat results. API-based reruns and a real Neo4j server integration were not performed. Numeric evidence replay does not claim verification of omitted historical graph database dumps or every omitted policy/tool trace.

Machine-readable results: `validation.json`. Dependency versions: `validation-environment.txt`.
