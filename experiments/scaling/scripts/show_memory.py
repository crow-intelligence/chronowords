"""Print the last record of a memory.jsonl file as one summary line."""

import json
import sys
from pathlib import Path


r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()[-1])
t = r["training_record"] or {}
pairs = (t.get("memory") or {}).get("pairs", {})
print(
    f"{r['label']:16} budget={r['max_memory']!s:6} {r['status']:13} "
    f"{r['seconds_total']:7.1f}s peak {r['peak_rss_mb'] / 1000:5.2f}GB "
    f"pairs cand/kept {pairs.get('candidates', '-')}/{pairs.get('kept', '-')} "
    f"nnz {t.get('ppmi_nnz')}"
)
