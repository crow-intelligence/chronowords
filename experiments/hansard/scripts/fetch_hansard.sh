#!/usr/bin/env bash
# Mirror the House of Commons debates XML (1919-today) published by
# TheyWorkForYou / ParlParse (mySociety), untouched, into data/raw/hansard/,
# and write a provenance manifest beside it.
#
# Licence: Contains Parliamentary information licensed under the Open
# Parliament Licence v3.0. See data/raw/README.md for attribution.
#
# Usage: experiments/hansard/scripts/fetch_hansard.sh   (~9 GB, all versions)
set -euo pipefail

cd "$(dirname "$0")/../../.."
SRC=data.theyworkforyou.com::parldata/scrapedxml/debates/
DEST=data/raw/hansard/debates
mkdir -p "$DEST"
rsync -a --stats "$SRC" "$DEST/"

python3 - "$DEST" "$SRC" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
dest, src = Path(sys.argv[1]), sys.argv[2]
files = sorted(p for p in dest.glob("debates*.xml"))
listing = "".join(f"{p.name}\t{p.stat().st_size}\n" for p in files)
manifest = {
    "source": f"rsync://{src}",
    "publisher": "TheyWorkForYou / ParlParse (mySociety)",
    "licence": "Open Parliament Licence v3.0",
    "files": len(files),
    "bytes": sum(p.stat().st_size for p in files),
    "listing_sha256": hashlib.sha256(listing.encode()).hexdigest(),
    "first": files[0].name if files else None,
    "last": files[-1].name if files else None,
    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
out = dest.parent / "debates.manifest.json"
out.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
