# export_task_specs.py  (adjust the import for your environment)
import json, re, unicodedata
from pathlib import Path
from task_spec import task_specs        # your list of dicts

def slugify(txt):
    t = unicodedata.normalize("NFKD", txt).encode("ascii","ignore").decode()
    t = re.sub(r"[^\w\s-]", "", t).strip().lower()
    return re.sub(r"[-\s]+", "_", t)

rows = []
for spec in task_specs:
    rows.append({
        "task"       : slugify(spec["name"]),
        "name"       : spec["name"],              # ← renamed
        "variables"  : json.dumps(spec["variables"]),
        "prompt"     : spec["prompt"],
        "levels"     : json.dumps(spec["levels"]),
        "cond_range" : json.dumps(spec.get("cond_range")),
        "dataset"    : spec["dataset"],           # e.g. "data/acs.parquet"
    })

Path("./task_specs.json").write_text(json.dumps(rows, indent=2))
print("✓ task_specs.json created with", len(rows), "tasks")
