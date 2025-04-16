# psi_c_ai_sdk/schema/snapshot.py

import os
import json
from datetime import datetime
from psi_c_ai_sdk.core.trace_context import get_trace_id

SNAPSHOT_DIR = "schema_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def save_schema_snapshot(schema: dict, label: str):
    """Save the current schema to a timestamped file labeled before/after."""
    trace_id = get_trace_id() or "no_trace"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{trace_id}_{label}_{timestamp}.json"
    path = os.path.join(SNAPSHOT_DIR, fname)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    return path
