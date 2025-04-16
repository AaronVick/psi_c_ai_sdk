# psi_c_ai_sdk/monitor/shared_schema_view.py

def export_shared_schema(merged_schema, path="shared_schema.graphml"):
    merged_schema.to_graphml(path)
    print(f"[ΨC] Shared schema exported to: {path}")
