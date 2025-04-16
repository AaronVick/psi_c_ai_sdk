# psi_c_ai_sdk/api/introspection_api.py

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI()

TRACE_LOG_PATH = "introspection_log.jsonl"
SNAPSHOT_DIR = "schema_snapshots"

@app.get("/introspect/trace")
def get_trace_log(limit: int = 100):
    if not os.path.exists(TRACE_LOG_PATH):
        raise HTTPException(status_code=404, detail="No introspection log found")
    with open(TRACE_LOG_PATH, "r") as f:
        lines = f.readlines()[-limit:]
    events = [json.loads(line) for line in lines]
    return JSONResponse(content=events)

@app.get("/introspect/snapshot/{filename}")
def get_snapshot_file(filename: str):
    path = os.path.join(SNAPSHOT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(path, media_type="application/json")
