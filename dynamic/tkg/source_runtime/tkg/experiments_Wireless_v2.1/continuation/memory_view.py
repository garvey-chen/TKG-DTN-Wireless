"""Lossless table encoding for model observations, never graph storage.

Shared fields apply to every row. A final ``extra`` column, when present,
contains optional fields that are absent from some records. This keeps absent
fields distinct from explicit nulls without repeating every common JSON key.
"""

from copy import deepcopy
import json


VIEW_VERSION = "continuation_memory_tables_v1"


def identical_json(left, right):
    """Exact JSON equality, including numeric representation; no rounding."""
    options = {"sort_keys": True, "ensure_ascii": False, "allow_nan": False}
    return json.dumps(left, **options) == json.dumps(right, **options)


def record_table(records, prefix):
    """Encode all records in order, with shared values stated exactly once.

    Only keys present in every record become table columns or shared fields.
    Optional keys are kept verbatim in each row's ``extra`` dictionary.
    """
    if not records:
        return {prefix + "_columns": [], prefix + "_rows": []}
    # The known event/change/user schemas have no field called extra. Fail
    # loudly if a future schema adds it rather than ambiguously merging data.
    if any("extra" in row for row in records):
        raise ValueError("The compact table reserves extra for optional fields")
    common = [key for key in records[0] if all(key in row for row in records)]
    shared = {key: deepcopy(records[0][key]) for key in common
              if all(identical_json(row[key], records[0][key]) for row in records[1:])}
    columns = [key for key in common if key not in shared]
    extras = [{key: deepcopy(value) for key, value in row.items() if key not in common}
              for row in records]
    has_extras = any(extras)
    rows = [[deepcopy(row[key]) for key in columns] + ([extra] if has_extras else [])
            for row, extra in zip(records, extras)]
    table = {prefix + "_columns": columns + (["extra"] if has_extras else []),
             prefix + "_rows": rows}
    if shared:
        table[prefix + "_shared"] = shared
    return table
